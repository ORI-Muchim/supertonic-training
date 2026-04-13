"""End-to-end TTS inference with Classifier-Free Guidance (CFG).

Runs the full PyTorch pipeline using trained checkpoints:
    text + voice_style (JSON) → wav (.wav file)

Unlike the shipped `py/helper.py`, this runner applies CFG — the paper-recommended
quality boost (typical cfg_scale=3). It uses the `VectorFieldCFG` wrapper so each
ODE step runs velocity TWICE (cond + uncond) and combines them.

Usage:
    python -m training.scripts.infer_cfg \
        --ae_ckpt  training/runs/ae_v1/ckpt_step00300000.pt \
        --stats    training/runs/ae_v1/cache/stats.pt \
        --ttl_ckpt training/runs/ttl_v1/ckpt_step00150000.pt \
        --dp_ckpt  training/runs/dp_v1/ckpt_step00003000.pt \
        --voice_style assets/voice_styles/F1.json \
        --text "안녕하세요, 이것은 CFG를 적용한 합성 결과입니다." \
        --lang ko --cfg_scale 3.0 --total_step 5 \
        --out out.wav
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "py"))

from training.models.ae_decoder import AEDecoder
from training.models.vector_field_cfg import load_cfg_from_ttl_checkpoint
from helper import UnicodeProcessor                        # type: ignore
from torch_text_encoder import TextEncoder                  # type: ignore
from torch_duration_predictor import DurationPredictor      # type: ignore


SAMPLE_RATE = 44100
HOP = 512
KC = 6
LDIM = 24
NORMALIZER_SCALE = 0.25


def _load_voice_style(path: str):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    style_ttl = np.array(d["style_ttl"]["data"], dtype=np.float32).reshape(d["style_ttl"]["dims"])
    style_dp  = np.array(d["style_dp"]["data"],  dtype=np.float32).reshape(d["style_dp"]["dims"])
    return torch.from_numpy(style_ttl), torch.from_numpy(style_dp)


def _load_decoder(ae_ckpt: str, stats_path: str, device: torch.device):
    """Build pure AEDecoder + latent_mean/std from AE ckpt + stats."""
    dec = AEDecoder().to(device)
    ae_sd = torch.load(ae_ckpt, map_location=device, weights_only=False)["ae"]
    dec_sd = {k.replace("decoder.", "", 1): v for k, v in ae_sd.items() if k.startswith("decoder.")}
    missing, unexpected = dec.load_state_dict(dec_sd, strict=False)
    if missing or unexpected:
        print(f"[warn] decoder load: missing={missing[:2]}  unexpected={unexpected[:2]}")
    dec.eval()
    s = torch.load(stats_path, map_location=device, weights_only=False)
    return dec, s["mean"].to(device), s["std"].to(device)


def _build_models(args, device):
    """Returns dict of (frozen, eval'd) modules."""
    ck_ttl = torch.load(args.ttl_ckpt, map_location=device, weights_only=False)
    ck_dp  = torch.load(args.dp_ckpt,  map_location=device, weights_only=False)

    te = TextEncoder().to(device); te.load_state_dict(ck_ttl["text_encoder"]); te.eval()
    dp = DurationPredictor().to(device); dp.load_state_dict(ck_dp["dp"]); dp.eval()
    vf_cfg = load_cfg_from_ttl_checkpoint(args.ttl_ckpt, device)

    decoder, latent_mean, latent_std = _load_decoder(args.ae_ckpt, args.stats, device)
    return {
        "text_encoder": te, "duration_predictor": dp,
        "vector_field_cfg": vf_cfg, "decoder": decoder,
        "latent_mean": latent_mean, "latent_std": latent_std,
    }


def _un_chunk_denormalize(z_ttl: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Inverse of cache_latents.prepare: [B, 144, L_ttl] → [B, 24, L_ae]."""
    B, C, L = z_ttl.shape
    z = z_ttl / NORMALIZER_SCALE
    z = z.reshape(B, LDIM, KC, L).permute(0, 1, 3, 2).contiguous().reshape(B, LDIM, L * KC)
    z = z * std.view(1, -1, 1) + mean.view(1, -1, 1)
    return z


def _sample_noisy_latent(duration_sec: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    """duration_sec [B] → (z_0 [B, 144, L_ttl], latent_mask [B, 1, L_ttl])."""
    B = duration_sec.shape[0]
    wav_len_max = duration_sec.max().item() * SAMPLE_RATE
    wav_lengths = (duration_sec * SAMPLE_RATE).long()
    chunk = HOP * KC
    L_ttl = int((wav_len_max + chunk - 1) // chunk)
    noisy = torch.randn(B, LDIM * KC, L_ttl, device=device)
    lat_mask = torch.zeros(B, 1, L_ttl, device=device)
    for i in range(B):
        L_i = int((wav_lengths[i] + chunk - 1) // chunk)
        lat_mask[i, 0, :L_i] = 1.0
    return noisy * lat_mask, lat_mask


def synthesize(
    text: str, lang: str,
    style_ttl: torch.Tensor, style_dp: torch.Tensor,
    models: dict, text_processor: UnicodeProcessor,
    cfg_scale: float = 3.0, total_step: int = 5, speed: float = 1.05,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, float]:
    """Returns (wav [n_samples] np.float32, duration_seconds)."""
    # 1) Tokenize
    text_ids_np, text_mask_np = text_processor([text], [lang])
    text_ids  = torch.from_numpy(text_ids_np).to(device)
    text_mask = torch.from_numpy(text_mask_np.astype("float32")).to(device)
    style_ttl = style_ttl.to(device)
    style_dp  = style_dp.to(device)

    with torch.no_grad():
        # 2) Duration predictor
        dur_sec = models["duration_predictor"](text_ids, style_dp, text_mask) / speed   # [1]

        # 3) Text encoder
        text_emb = models["text_encoder"](text_ids, style_ttl, text_mask)                # [1, 256, T]

        # 4) Initial noise
        z_t, lat_mask = _sample_noisy_latent(dur_sec, device)

        # 5) Flow matching Euler loop with CFG
        total_np = torch.tensor([total_step] * 1, dtype=torch.float32, device=device)
        cfg_np   = torch.tensor([cfg_scale] * 1, dtype=torch.float32, device=device)
        for step in range(total_step):
            current = torch.tensor([step] * 1, dtype=torch.float32, device=device)
            z_t = models["vector_field_cfg"](
                z_t, text_emb, style_ttl, lat_mask, text_mask,
                current, total_np, cfg_np,
            )

        # 6) Decode: un-chunk + de-normalize + AE decoder
        z_ae = _un_chunk_denormalize(z_t, models["latent_mean"], models["latent_std"])
        wav = models["decoder"](z_ae)     # [1, T_ae * hop]

    wav_np = wav[0].cpu().numpy().astype("float32")
    dur = float(dur_sec.item())
    wav_np = wav_np[: int(dur * SAMPLE_RATE)]
    return wav_np, dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae_ckpt",  required=True)
    ap.add_argument("--stats",    required=True)
    ap.add_argument("--ttl_ckpt", required=True)
    ap.add_argument("--dp_ckpt",  required=True)
    ap.add_argument("--voice_style", required=True, help="voice_styles/*.json")
    ap.add_argument("--text",     required=True)
    ap.add_argument("--lang",     default="ko")
    ap.add_argument("--cfg_scale", type=float, default=3.0,
                    help="CFG guidance weight (0=disable, paper recommends 3)")
    ap.add_argument("--total_step", type=int, default=5)
    ap.add_argument("--speed",    type=float, default=1.05)
    ap.add_argument("--out",      default="out.wav")
    ap.add_argument("--unicode_indexer", default=str(ROOT / "assets" / "onnx" / "unicode_indexer.json"))
    ap.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[info] device={device}  cfg_scale={args.cfg_scale}  total_step={args.total_step}")

    text_processor = UnicodeProcessor(args.unicode_indexer)
    models = _build_models(args, device)
    style_ttl, style_dp = _load_voice_style(args.voice_style)

    wav_np, dur = synthesize(
        args.text, args.lang, style_ttl, style_dp, models, text_processor,
        cfg_scale=args.cfg_scale, total_step=args.total_step, speed=args.speed, device=device,
    )
    sf.write(args.out, wav_np, SAMPLE_RATE)
    print(f"[done] duration={dur:.2f}s  samples={len(wav_np)}  wrote → {args.out}")


if __name__ == "__main__":
    main()
