"""End-to-end synth for paper-faithful Stage-2 (700k TTL) — optional DP.

Loads:
  - AE decoder from AE ckpt
  - paper-faithful TextEncoderPaper, StyleEncoderTTLPaper, VectorField(dim=256, ...)
    + uncond_masker tokens from TTL ckpt
  - optionally: DP ckpt (StyleEncoderDPPaper + DurationPredictor) for predicted duration

Reference style:
  - Picks one utterance from the AE latent cache (default: idx 0)
  - Runs StyleEncoderTTLPaper over its full latent → [1, 50, 128] style_ttl

Duration:
  - With --dp_ckpt: DP predicts utterance duration.
  - Without --dp_ckpt: heuristic char_count × 0.18 s (KO) or --duration_sec.

Inference defaults:
  - cfg_scale = 1.0  — paper recommends 3.0, but this single-speaker model has a
    tighter latent distribution and cfg=3 causes hard clipping (3.6% saturation
    in test runs; see analysis/synth_blowup_analysis.py).
  - total_step = 16 — better Euler ODE truncation than the 5-step default.
  - Single joint uncond replacement (paper p_uncond=0.05).

Usage:
  python -m training.scripts.synth_ttl_paper \
    --ae_ckpt  training/runs/ae_paper_audit_crop1s_mean/ckpt_step01500000.pt \
    --cache_dir training/runs/ae_paper_audit_crop1s_mean/cache \
    --ttl_ckpt training/runs/ttl_paper_rope_b32a2/ckpt_step00700000.pt \
    --dp_ckpt  training/runs/dp_paper_rope_3k/ckpt_step00003000.pt \
    --text "안녕하세요 슈퍼토닉 페이퍼 충실 합성 결과입니다" \
    --out out_ttl_paper.wav
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
from training.models.style_encoder import StyleEncoderTTLPaper, StyleEncoderDPPaper
from training.data.ttl_dataset import invert_ttl_latent, TTL_NORMALIZER_SCALE
from helper import UnicodeProcessor                          # type: ignore
from torch_text_encoder import TextEncoderPaper              # type: ignore
from torch_vector_estimator import VectorField               # type: ignore
from torch_duration_predictor import DurationPredictor       # type: ignore


SAMPLE_RATE = 44100
HOP = 512
KC = 6
LDIM = 24


def _load_decoder(ae_ckpt: str, device: torch.device) -> AEDecoder:
    dec = AEDecoder().to(device)
    ae_sd = torch.load(ae_ckpt, map_location=device, weights_only=False)["ae"]
    dec_sd = {k.replace("decoder.", "", 1): v for k, v in ae_sd.items() if k.startswith("decoder.")}
    missing, unexpected = dec.load_state_dict(dec_sd, strict=False)
    if missing or unexpected:
        print(f"[warn] decoder load: missing={missing[:2]} unexpected={unexpected[:2]}")
    dec.eval()
    return dec


def _build_dp_models(dp_ckpt: str, device: torch.device):
    """Load DP + StyleEncoderDPPaper from a train_dp.py checkpoint.
    Default-trained DP uses shipped-authority 192-dim estimator (DurationPredictor).
    attn_type ('rope' or 'relpos') is read from the checkpoint's saved cfg.
    """
    ck = torch.load(dp_ckpt, map_location=device, weights_only=False)
    attn_type = ck.get("cfg", {}).get("attn_type", "relpos")
    print(f"[info] DP attn_type from ckpt: {attn_type}")
    dp = DurationPredictor(attn_type=attn_type).to(device)   # shipped 192-dim path
    se_dp = StyleEncoderDPPaper().to(device)                 # paper out_scale=1.0, [B,8,16]
    dp.load_state_dict(ck["dp"])
    se_dp.load_state_dict(ck["style_enc"])
    dp.eval(); se_dp.eval()
    return dp, se_dp


def _build_paper_models(ttl_ckpt: str, device: torch.device):
    ck = torch.load(ttl_ckpt, map_location=device, weights_only=False)
    te = TextEncoderPaper(style_dim=128).to(device)
    se = StyleEncoderTTLPaper().to(device)
    vf = VectorField(
        dim=256, latent_dim=144, n_outer=4, time_dim=64,
        inter=1024, ksz=5, text_dim=128, style_dim=128,
        learn_style_prototype=False,
    ).to(device)
    te.load_state_dict(ck["text_encoder"])
    se.load_state_dict(ck["style_encoder"])
    vf.load_state_dict(ck["vector_field"])
    um = ck["uncond_masker"]
    uncond_text  = um["uncond_text"].to(device)    # [1, 128, 1]
    uncond_style = um["uncond_style"].to(device)   # [1, 50, 128]
    te.eval(); se.eval(); vf.eval()
    return te, se, vf, uncond_text, uncond_style


def _load_ref_latent(cache_dir: Path, ref_idx: int, device: torch.device):
    with open(cache_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    m = manifest[ref_idx]
    z_ae = torch.load(cache_dir / m["latent_path"], weights_only=False).float().to(device)  # [24, T]
    stats = torch.load(cache_dir / "stats.pt", weights_only=False)
    mean = stats["mean"].float().to(device)
    std  = stats["std"].float().to(device)
    print(f"[info] ref idx={ref_idx} text='{m['text_raw'][:30]}...' dur={m['duration']:.2f}s T_ae={z_ae.shape[-1]}")
    return z_ae, mean, std, m


def _make_z0_and_mask(L_ttl: int, device: torch.device):
    z_0 = torch.randn(1, LDIM * KC, L_ttl, device=device)
    lat_mask = torch.ones(1, 1, L_ttl, device=device)
    return z_0 * lat_mask, lat_mask


@torch.no_grad()
def synth(
    text: str, lang: str,
    z_ae_ref: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
    te: TextEncoderPaper, se: StyleEncoderTTLPaper, vf: VectorField,
    uncond_text: torch.Tensor, uncond_style: torch.Tensor,
    decoder: AEDecoder, text_processor: UnicodeProcessor,
    duration_sec: float, cfg_scale: float, total_step: int,
    device: torch.device,
    dp: DurationPredictor | None = None, se_dp: StyleEncoderDPPaper | None = None,
):
    # 1) tokenize
    text_ids_np, text_mask_np = text_processor([text], [lang])
    text_ids  = torch.from_numpy(text_ids_np).to(device)
    text_mask = torch.from_numpy(text_mask_np.astype("float32")).to(device)

    # 2) reference style: z_ae_ref [24, T] -> z-score -> [1, 24, T]
    ref = (z_ae_ref - mean.view(-1, 1)) / std.view(-1, 1)
    ref = ref.unsqueeze(0)
    ref_mask = torch.ones(1, 1, ref.shape[-1], device=device)
    style_ttl = se(ref, ref_mask)                                     # [1, 50, 128]

    # 2b) DP-predicted duration (overrides duration_sec when DP supplied)
    if dp is not None and se_dp is not None:
        style_dp = se_dp(ref, ref_mask)                               # [1, 8, 16]
        dur_pred = dp(text_ids, style_dp, text_mask).item()
        print(f"[info] DP-predicted dur={dur_pred:.2f}s (heuristic was {duration_sec:.2f}s)")
        duration_sec = float(dur_pred)

    # 3) text emb
    text_emb = te(text_ids, style_ttl, text_mask)                     # [1, 128, T_text]

    # 4) target latent length from duration
    chunk = HOP * KC                                                  # 3072 samples per TTL frame
    L_ttl = max(1, int(np.ceil(duration_sec * SAMPLE_RATE / chunk)))
    z_t, lat_mask = _make_z0_and_mask(L_ttl, device)
    print(f"[info] target dur={duration_sec:.2f}s L_ttl={L_ttl} cfg={cfg_scale} steps={total_step}")

    # 5) Euler ODE loop with CFG (paper σ_min=1e-8 → effectively z_t = (1-t)z_0 + t z_1)
    style_proto = te.reference_key
    text_u  = uncond_text.expand(1, -1, text_emb.shape[-1])           # [1,128,T_text]
    style_u = uncond_style                                            # [1,50,128]
    for step in range(total_step):
        t_norm = torch.tensor([step / total_step], device=device)
        v_cond = vf.velocity(z_t, text_emb, style_ttl, lat_mask, text_mask, t_norm,
                             style_prototype=style_proto)
        if cfg_scale != 0.0:
            v_unc = vf.velocity(z_t, text_u, style_u, lat_mask, text_mask, t_norm,
                                style_prototype=style_proto)
            v = (1.0 + cfg_scale) * v_cond - cfg_scale * v_unc
        else:
            v = v_cond
        dt = 1.0 / total_step
        z_t = (z_t + dt * v) * lat_mask

    # 6) invert chunk-compress + de-normalize -> AE latent [1, 24, T_ae]
    z_ttl = z_t.squeeze(0)                                            # [144, L_ttl]
    z_ae = invert_ttl_latent(z_ttl, mean, std, scale=TTL_NORMALIZER_SCALE, kc=KC).unsqueeze(0)

    # 7) decode
    wav = decoder(z_ae)                                               # [1, T_ae*hop]
    wav = wav.squeeze(0).cpu().numpy().astype("float32")
    n = int(round(duration_sec * SAMPLE_RATE))
    wav = wav[: n]
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae_ckpt",  required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ttl_ckpt", required=True)
    ap.add_argument("--text",     required=True)
    ap.add_argument("--lang",     default="ko")
    ap.add_argument("--ref_idx",  type=int, default=0)
    ap.add_argument("--duration_sec", type=float, default=None,
                    help="if omitted: heuristic char_count*0.07s (KO)")
    ap.add_argument("--cfg_scale", type=float, default=1.0,
                    help="paper recommends 3.0 but this model's tighter latent distribution causes "
                         "hard clipping at cfg>=3 (e.g. 3.6%% samples saturated at cfg=3); 1.0 is empirical sweet spot")
    ap.add_argument("--total_step", type=int, default=16,
                    help="paper Euler steps; 5 is fast/low-quality, 16 gives better ODE truncation")
    ap.add_argument("--dp_ckpt", default=None,
                    help="if provided, DP predicts duration (overrides --duration_sec/heuristic)")
    ap.add_argument("--out", default="out_ttl_paper.wav")
    ap.add_argument("--unicode_indexer", default=str(ROOT / "assets" / "onnx" / "unicode_indexer.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"[info] device={device}")

    text_processor = UnicodeProcessor(args.unicode_indexer)
    decoder = _load_decoder(args.ae_ckpt, device)
    te, se, vf, uncond_text, uncond_style = _build_paper_models(args.ttl_ckpt, device)
    z_ae_ref, mean, std, ref_meta = _load_ref_latent(Path(args.cache_dir), args.ref_idx, device)

    dp = se_dp = None
    if args.dp_ckpt is not None:
        dp, se_dp = _build_dp_models(args.dp_ckpt, device)
        print(f"[info] DP loaded from {args.dp_ckpt}")

    if args.duration_sec is None:
        char_count = sum(1 for c in args.text if not c.isspace())
        dur = max(1.0, char_count * 0.18)
        print(f"[info] heuristic duration (fallback): {char_count} chars × 0.18s = {dur:.2f}s")
    else:
        dur = args.duration_sec

    wav = synth(
        args.text, args.lang,
        z_ae_ref, mean, std,
        te, se, vf, uncond_text, uncond_style,
        decoder, text_processor,
        duration_sec=dur, cfg_scale=args.cfg_scale, total_step=args.total_step,
        device=device,
        dp=dp, se_dp=se_dp,
    )
    sf.write(args.out, wav, SAMPLE_RATE)
    print(f"[done] wrote -> {args.out}  duration={len(wav)/SAMPLE_RATE:.2f}s  samples={len(wav)}")


if __name__ == "__main__":
    main()
