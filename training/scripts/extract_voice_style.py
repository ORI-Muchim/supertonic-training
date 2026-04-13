"""Extract a voice style from a reference audio clip.

Input:  reference wav (any SR, mono or stereo)
Output: voice_styles/{name}.json with style_ttl [1,50,256] + style_dp [1,8,16],
        matching the format of shipped assets/voice_styles/*.json

Usage:
    # From trained checkpoints:
    python -m training.scripts.extract_voice_style \
        --wav my_voice.wav \
        --ae_ckpt training/runs/ae_v1/ckpt_step00300000.pt \
        --ttl_ckpt training/runs/ttl_v1/ckpt_step00150000.pt \
        --dp_ckpt  training/runs/dp_v1/ckpt_step00003000.pt \
        --out my_voice.json \
        --name "Jane"

    # The TTL and DP checkpoints contain their own style_encoders trained alongside
    # the text_encoder / duration_predictor respectively.

Inference note:
    style encoders are NOT in the released ONNX — they only exist at training time.
    This script is the replacement: run once per speaker to produce a JSON that
    `py/helper.py` / other language runtimes can consume.
"""
from __future__ import annotations
import os, sys, json, argparse, datetime
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.data.spectrogram import SpecProcessor
from training.models.ae_encoder import AEEncoder
from training.models.style_encoder import StyleEncoderTTL, StyleEncoderDP


TARGET_SR = 44100


def _load_and_resample_mono(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    orig_sr = int(sr)
    if orig_sr != TARGET_SR:
        try:
            import librosa
        except ImportError:
            raise RuntimeError(
                f"Reference audio is {orig_sr} Hz but target is {TARGET_SR}. "
                "Install librosa for resampling: `pip install librosa`."
            )
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=TARGET_SR)
    # Peak-normalize mildly so encoder sees reasonable range
    peak = float(np.abs(wav).max())
    if peak > 0.99:
        wav = wav * (0.99 / peak)
    return wav.astype("float32"), orig_sr


def _prefix_strip(sd: dict, prefix: str) -> dict:
    """If state_dict keys all start with `prefix.`, strip it."""
    if all(k.startswith(prefix + ".") for k in sd.keys()):
        return {k[len(prefix)+1:]: v for k, v in sd.items()}
    return sd


def _load_encoder_and_style(
    ae_ckpt: str, ttl_ckpt: str, dp_ckpt: str, device: torch.device,
):
    """Load AE encoder from ae_ckpt, StyleEncoderTTL from ttl_ckpt,
    StyleEncoderDP from dp_ckpt."""
    # AE encoder
    ae_sd = torch.load(str(ae_ckpt), map_location=device, weights_only=False)["ae"]
    enc = AEEncoder().to(device)
    enc_sd = {k.replace("encoder.", "", 1): v for k, v in ae_sd.items() if k.startswith("encoder.")}
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    if missing or unexpected:
        print(f"[warn] AE encoder load: missing={missing[:2]}...  unexpected={unexpected[:2]}...")
    enc.eval()

    # TTL style encoder (saved under 'style_encoder' key in train_ttl.py)
    se_ttl = StyleEncoderTTL().to(device)
    ttl_ck = torch.load(str(ttl_ckpt), map_location=device, weights_only=False)
    se_ttl.load_state_dict(ttl_ck["style_encoder"])
    se_ttl.eval()

    # DP style encoder (saved under 'style_enc' key in train_dp.py)
    se_dp = StyleEncoderDP().to(device)
    dp_ck = torch.load(str(dp_ckpt), map_location=device, weights_only=False)
    se_dp.load_state_dict(dp_ck["style_enc"])
    se_dp.eval()

    return enc, se_ttl, se_dp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav",      type=str, required=True, help="reference audio file")
    ap.add_argument("--ae_ckpt",  type=str, required=True)
    ap.add_argument("--ttl_ckpt", type=str, required=True)
    ap.add_argument("--dp_ckpt",  type=str, required=True)
    ap.add_argument("--out",      type=str, required=True, help="output JSON path")
    ap.add_argument("--name",     type=str, default=None,
                    help="speaker name stored in metadata (default: basename of --wav)")
    ap.add_argument("--device",   type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    name = args.name or Path(args.wav).stem

    # ---- load reference wav ----
    print(f"[info] loading reference {args.wav}")
    wav_np, orig_sr = _load_and_resample_mono(args.wav)
    dur = len(wav_np) / TARGET_SR
    print(f"  original SR: {orig_sr} Hz  →  resampled to {TARGET_SR} Hz   duration: {dur:.2f} s")
    if dur < 1.0:
        print(f"  [warn] reference is very short ({dur:.2f}s). 3-10 s is typically recommended.")

    wav = torch.from_numpy(wav_np).unsqueeze(0).to(device)  # [1, T_samples]

    # ---- models ----
    spec = SpecProcessor().to(device)
    enc, se_ttl, se_dp = _load_encoder_and_style(args.ae_ckpt, args.ttl_ckpt, args.dp_ckpt, device)

    # ---- forward ----
    with torch.no_grad():
        feats  = spec(wav)                # [1, 1253, T_frames]
        latent = enc(feats)                # [1, 24, T_frames]
        style_ttl = se_ttl(latent)         # [1, 50, 256]
        style_dp  = se_dp(latent)          # [1, 8, 16]

    style_ttl_np = style_ttl.cpu().numpy().astype("float32")
    style_dp_np  = style_dp.cpu().numpy().astype("float32")

    # ---- write JSON ----
    out_json = {
        "style_ttl": {
            "data": style_ttl_np.tolist(),
            "dims": list(style_ttl_np.shape),
            "type": "float32",
        },
        "style_dp": {
            "data": style_dp_np.tolist(),
            "dims": list(style_dp_np.shape),
            "type": "float32",
        },
        "metadata": {
            "source_file": Path(args.wav).name,
            "source_sample_rate": orig_sr,
            "target_sample_rate": TARGET_SR,
            "extracted_at": datetime.datetime.utcnow().isoformat() + "Z",
            "speaker_name": name,
            "duration_seconds": round(dur, 3),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=1)

    print(f"[done] style_ttl shape: {style_ttl_np.shape}  "
          f"norm: {np.linalg.norm(style_ttl_np):.2f}")
    print(f"       style_dp  shape: {style_dp_np.shape}  "
          f"norm: {np.linalg.norm(style_dp_np):.2f}")
    print(f"       saved  → {out_path}")
    print(f"       use with: python py/example_onnx.py --voice-style {out_path}")


if __name__ == "__main__":
    main()
