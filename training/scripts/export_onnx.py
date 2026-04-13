"""Export trained models to ONNX matching the released `assets/onnx/*.onnx` contracts.

Produces:
    out_dir/
        duration_predictor.onnx     # from trained DP + style_encoder (DP variant)
        text_encoder.onnx           # from trained text_encoder (style_ttl used as input)
        vector_estimator.onnx       # from trained vector_field (Euler step included)
        vocoder.onnx                # AEDecoder wrapped with un-chunk + de-normalize
        tts.json                    # config snapshot (may be hand-edited later)
        unicode_indexer.json        # copied from assets/onnx/ (we don't retrain tokenizer)

Note: text_encoder / vector_estimator / duration_predictor are exported with their
FORWARD methods as-is. For vocoder we need to BAKE the TTL-specific preprocessing
(un-chunk 6×, z-score de-normalization, division by normalizer_scale=0.25) into the
ONNX graph since the released vocoder.onnx has them.

Usage:
    python -m training.scripts.export_onnx \
        --ae_ckpt training/runs/ae_v1/ckpt_step00300000.pt \
        --stats   training/runs/ae_v1/cache/stats.pt \
        --ttl_ckpt training/runs/ttl_v1/ckpt_step00150000.pt \
        --dp_ckpt  training/runs/dp_v1/ckpt_step00003000.pt \
        --out_dir  training/runs/exported/

Caveats:
  • `assets/onnx/unicode_indexer.json` is reused verbatim (we don't retrain the 163-token vocab).
  • style_encoder (both TTL and DP) is NOT exported — at inference time style is provided
    as a precomputed JSON via `extract_voice_style.py`.
"""
from __future__ import annotations
import os, sys, json, argparse, shutil
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from training.models.ae_decoder import AEDecoder
from training.models.vector_field_cfg import VectorFieldCFG, load_cfg_from_ttl_checkpoint
# Verified inference modules from analysis/
from torch_text_encoder import TextEncoder                       # type: ignore
from torch_duration_predictor import DurationPredictor           # type: ignore
from torch_vector_estimator import VectorField                   # type: ignore

ASSETS_ONNX = ROOT / "assets" / "onnx"


# ---------------------------------------------------------------------
# Vocoder wrapper: AEDecoder + TTL-specific preprocessing (matches shipped vocoder.onnx)
# ---------------------------------------------------------------------
class VocoderExportWrapper(nn.Module):
    """Bakes un-chunk 6× + de-normalize + /normalizer_scale into one ONNX module.

    Input:  latent [B, 144, L]   (TTL-space: z-scored & scaled by 0.25 & chunk-compressed)
    Output: wav_tts [B, L*6*512] (float, model's predicted waveform)
    """
    def __init__(
        self,
        ae_decoder: AEDecoder,
        latent_mean: torch.Tensor,      # [24]
        latent_std:  torch.Tensor,      # [24]
        normalizer_scale: float = 0.25, # tts.json: ttl.normalizer.scale
        kc: int = 6,
        ldim: int = 24,
    ):
        super().__init__()
        self.decoder = ae_decoder
        self.kc = kc
        self.ldim = ldim
        self.register_buffer("normalizer_scale", torch.tensor(float(normalizer_scale)))
        self.register_buffer("latent_mean", latent_mean.view(1, ldim, 1).float())
        self.register_buffer("latent_std",  latent_std.view(1, ldim, 1).float())

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B = latent.shape[0]; L = latent.shape[-1]
        x = latent / self.normalizer_scale                                          # de-scale
        x = x.reshape(B, self.ldim, self.kc, L)                                     # [B,24,6,L]
        x = x.permute(0, 1, 3, 2).contiguous()                                      # [B,24,L,6]
        x = x.reshape(B, self.ldim, L * self.kc)                                    # [B,24,6L]
        x = x * self.latent_std + self.latent_mean                                  # de-normalize
        wav = self.decoder(x)                                                       # [B, 6L*512]
        return wav


# ---------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------
def _load_state(path: str | Path, key: str, device: torch.device):
    ck = torch.load(str(path), map_location=device, weights_only=False)
    assert key in ck, f"key '{key}' not found in checkpoint {path} (keys: {list(ck.keys())})"
    return ck[key]


def export_vocoder(
    ae_ckpt: str | Path,
    stats: str | Path,
    out_path: Path,
    device: torch.device,
    opset: int = 19,
):
    print(f"[vocoder] building...")
    # AE decoder only (from the AE checkpoint)
    # Our SpeechAutoencoder stores decoder as `.decoder`. Load full state then extract.
    # But AE checkpoint saves under key 'ae' = SpeechAutoencoder.state_dict().
    ae_sd = _load_state(ae_ckpt, "ae", device)
    dec = AEDecoder().to(device)
    # Keys in ae_sd start with "decoder." — strip that prefix
    dec_sd = {k.replace("decoder.", "", 1): v for k, v in ae_sd.items() if k.startswith("decoder.")}
    missing, unexpected = dec.load_state_dict(dec_sd, strict=False)
    if missing or unexpected:
        print(f"  warn: missing={missing[:3]}... unexpected={unexpected[:3]}...")
    dec.eval()

    s = torch.load(str(stats), map_location=device, weights_only=False)
    mean, std = s["mean"], s["std"]
    print(f"  mean range [{mean.min():.3f}, {mean.max():.3f}]   std range [{std.min():.3f}, {std.max():.3f}]")

    wrap = VocoderExportWrapper(dec, mean, std).to(device).eval()

    # Dummy input: B=2, L=17 (TTL frames). 17*6*512 = 52224 samples out.
    dummy = torch.randn(2, 144, 17, device=device)
    print(f"  dummy check → out shape: {tuple(wrap(dummy).shape)}")

    torch.onnx.export(
        wrap, (dummy,), str(out_path),
        input_names=["latent"], output_names=["wav_tts"],
        dynamic_axes={"latent": {0: "batch_size", 2: "latent_length"},
                      "wav_tts": {0: "batch_size", 1: "wav_length"}},
        opset_version=opset, do_constant_folding=True,
    )
    print(f"[vocoder] saved → {out_path}")


def export_duration_predictor(
    dp_ckpt: str | Path,
    out_path: Path,
    device: torch.device,
    opset: int = 19,
):
    print(f"[dp] building...")
    dp = DurationPredictor().to(device)
    sd = _load_state(dp_ckpt, "dp", device)
    dp.load_state_dict(sd)
    dp.eval()

    dummy_text_ids  = torch.zeros(2, 30, dtype=torch.long, device=device)
    dummy_style_dp  = torch.randn(2, 8, 16, device=device)
    dummy_text_mask = torch.ones(2, 1, 30, device=device)
    print(f"  dummy check → out shape: {tuple(dp(dummy_text_ids, dummy_style_dp, dummy_text_mask).shape)}")

    torch.onnx.export(
        dp, (dummy_text_ids, dummy_style_dp, dummy_text_mask), str(out_path),
        input_names=["text_ids", "style_dp", "text_mask"],
        output_names=["duration"],
        dynamic_axes={
            "text_ids":  {0: "batch_size", 1: "text_length"},
            "style_dp":  {0: "batch_size"},
            "text_mask": {0: "batch_size", 2: "text_length"},
            "duration":  {0: "batch_size"},
        },
        opset_version=opset, do_constant_folding=True,
    )
    print(f"[dp] saved → {out_path}")


def export_text_encoder(
    ttl_ckpt: str | Path,
    out_path: Path,
    device: torch.device,
    opset: int = 19,
):
    print(f"[text_encoder] building...")
    te = TextEncoder().to(device)
    sd = _load_state(ttl_ckpt, "text_encoder", device)
    te.load_state_dict(sd)
    te.eval()

    dummy_text_ids   = torch.zeros(2, 30, dtype=torch.long, device=device)
    dummy_style_ttl  = torch.randn(2, 50, 256, device=device)
    dummy_text_mask  = torch.ones(2, 1, 30, device=device)
    print(f"  dummy check → out shape: {tuple(te(dummy_text_ids, dummy_style_ttl, dummy_text_mask).shape)}")

    torch.onnx.export(
        te, (dummy_text_ids, dummy_style_ttl, dummy_text_mask), str(out_path),
        input_names=["text_ids", "style_ttl", "text_mask"],
        output_names=["text_emb"],
        dynamic_axes={
            "text_ids":  {0: "batch_size", 1: "text_length"},
            "style_ttl": {0: "batch_size"},
            "text_mask": {0: "batch_size", 2: "text_length"},
            "text_emb":  {0: "batch_size", 2: "text_length"},
        },
        opset_version=opset, do_constant_folding=True,
    )
    print(f"[text_encoder] saved → {out_path}")


def export_vector_estimator(
    ttl_ckpt: str | Path,
    out_path: Path,
    device: torch.device,
    opset: int = 19,
):
    print(f"[vector_estimator] building...")
    vf = VectorField().to(device)
    sd = _load_state(ttl_ckpt, "vector_field", device)
    vf.load_state_dict(sd)
    vf.eval()

    # dummy input
    dummy_noisy  = torch.randn(2, 144, 17, device=device)
    dummy_te     = torch.randn(2, 256, 25, device=device)
    dummy_sttl   = torch.randn(2, 50, 256, device=device)
    dummy_lm     = torch.ones(2, 1, 17, device=device)
    dummy_tm     = torch.ones(2, 1, 25, device=device)
    dummy_cs     = torch.tensor([1.0, 2.0], device=device)
    dummy_ts     = torch.tensor([5.0, 5.0], device=device)
    print(f"  dummy check → out shape: {tuple(vf(dummy_noisy, dummy_te, dummy_sttl, dummy_lm, dummy_tm, dummy_cs, dummy_ts).shape)}")

    torch.onnx.export(
        vf,
        (dummy_noisy, dummy_te, dummy_sttl, dummy_lm, dummy_tm, dummy_cs, dummy_ts),
        str(out_path),
        input_names=["noisy_latent", "text_emb", "style_ttl",
                     "latent_mask", "text_mask", "current_step", "total_step"],
        output_names=["denoised_latent"],
        dynamic_axes={
            "noisy_latent":   {0: "batch_size", 2: "latent_length"},
            "text_emb":       {0: "batch_size", 2: "text_length"},
            "style_ttl":      {0: "batch_size"},
            "latent_mask":    {0: "batch_size", 2: "latent_length"},
            "text_mask":      {0: "batch_size", 2: "text_length"},
            "current_step":   {0: "batch_size"},
            "total_step":     {0: "batch_size"},
            "denoised_latent":{0: "batch_size", 2: "latent_length"},
        },
        opset_version=opset, do_constant_folding=True,
    )
    print(f"[vector_estimator] saved → {out_path}")


def export_vector_estimator_cfg(
    ttl_ckpt: str | Path,
    out_path: Path,
    device: torch.device,
    opset: int = 19,
):
    """Export CFG-aware vector_estimator.  Takes extra `cfg_scale [B]` input.
    When cfg_scale=0 the output matches the non-CFG vector_estimator exactly.
    When cfg_scale > 0 it applies classifier-free guidance (paper recommends 3.0)."""
    print(f"[vector_estimator_cfg] building...")
    cfg_mod = load_cfg_from_ttl_checkpoint(ttl_ckpt, device)

    # dummy input
    B = 2
    dummy_noisy  = torch.randn(B, 144, 17, device=device)
    dummy_te     = torch.randn(B, 256, 25, device=device)
    dummy_sttl   = torch.randn(B, 50, 256, device=device)
    dummy_lm     = torch.ones(B, 1, 17, device=device)
    dummy_tm     = torch.ones(B, 1, 25, device=device)
    dummy_cs     = torch.tensor([1.0, 2.0], device=device)
    dummy_ts     = torch.tensor([5.0, 5.0], device=device)
    dummy_cfg    = torch.tensor([3.0, 3.0], device=device)
    print(f"  dummy check → out shape: {tuple(cfg_mod(dummy_noisy, dummy_te, dummy_sttl, dummy_lm, dummy_tm, dummy_cs, dummy_ts, dummy_cfg).shape)}")

    torch.onnx.export(
        cfg_mod,
        (dummy_noisy, dummy_te, dummy_sttl, dummy_lm, dummy_tm, dummy_cs, dummy_ts, dummy_cfg),
        str(out_path),
        input_names=["noisy_latent", "text_emb", "style_ttl",
                     "latent_mask", "text_mask", "current_step", "total_step", "cfg_scale"],
        output_names=["denoised_latent"],
        dynamic_axes={
            "noisy_latent":   {0: "batch_size", 2: "latent_length"},
            "text_emb":       {0: "batch_size", 2: "text_length"},
            "style_ttl":      {0: "batch_size"},
            "latent_mask":    {0: "batch_size", 2: "latent_length"},
            "text_mask":      {0: "batch_size", 2: "text_length"},
            "current_step":   {0: "batch_size"},
            "total_step":     {0: "batch_size"},
            "cfg_scale":      {0: "batch_size"},
            "denoised_latent":{0: "batch_size", 2: "latent_length"},
        },
        opset_version=opset, do_constant_folding=True,
    )
    print(f"[vector_estimator_cfg] saved → {out_path}")


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae_ckpt", type=str, required=True, help="AE checkpoint")
    ap.add_argument("--stats",   type=str, required=True, help="stats.pt from cache_latents.py")
    ap.add_argument("--ttl_ckpt", type=str, required=True, help="TTL checkpoint (text_encoder + vector_field)")
    ap.add_argument("--dp_ckpt",  type=str, required=True, help="DP checkpoint")
    ap.add_argument("--out_dir",  type=str, required=True)
    ap.add_argument("--opset",    type=int, default=19)
    ap.add_argument("--device",   type=str, default="cpu")  # CPU export is safer & portable
    ap.add_argument("--no_cfg",   action="store_true",
                    help="Skip exporting the CFG-aware vector_estimator variant.")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    with torch.no_grad():
        export_vocoder(args.ae_ckpt, args.stats, out / "vocoder.onnx",                device, args.opset)
        export_duration_predictor(args.dp_ckpt, out / "duration_predictor.onnx",      device, args.opset)
        export_text_encoder(args.ttl_ckpt, out / "text_encoder.onnx",                 device, args.opset)
        export_vector_estimator(args.ttl_ckpt, out / "vector_estimator.onnx",         device, args.opset)
        if not args.no_cfg:
            try:
                export_vector_estimator_cfg(args.ttl_ckpt, out / "vector_estimator_cfg.onnx",
                                             device, args.opset)
            except KeyError as e:
                print(f"[vector_estimator_cfg] SKIPPED: checkpoint missing uncond_masker ({e}). "
                      "Re-train TTL with uncond_masker (default in train_ttl.py) to enable CFG export.")

    # Copy the unicode_indexer verbatim (we reuse the released 163-token vocab)
    src = ASSETS_ONNX / "unicode_indexer.json"
    if src.exists():
        shutil.copy2(src, out / "unicode_indexer.json")
        print(f"[unicode_indexer] copied from {src}")

    # Write a minimal tts.json matching the released format (for helper.py to load)
    tts_json = {
        "tts_version": "custom-kss",
        "split": "kss-ko",
        "ttl_ckpt_path": str(args.ttl_ckpt),
        "dp_ckpt_path":  str(args.dp_ckpt),
        "ae_ckpt_path":  str(args.ae_ckpt),
        "ttl_train": "KSS single-speaker ko",
        "dp_train":  "KSS single-speaker ko",
        "ae_train":  "KSS single-speaker ko",
        "ae":  {"sample_rate": 44100, "base_chunk_size": 512,
                "chunk_compress_factor": 1, "ldim": 24, "n_delay": 0},
        "ttl": {"latent_dim": 24, "chunk_compress_factor": 6,
                "normalizer": {"scale": 0.25}},
    }
    with open(out / "tts.json", "w") as f:
        json.dump(tts_json, f, indent=2)
    print(f"[tts.json] wrote minimal config → {out/'tts.json'}")

    print(f"\n[done] all models exported under {out}")
    print("      To use with py/helper.py:  python py/example_onnx.py --onnx-dir", out)


if __name__ == "__main__":
    main()
