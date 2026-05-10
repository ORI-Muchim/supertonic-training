"""Evaluate AE checkpoints on deterministic KSS crops.

This is intentionally small: it compares checkpoints with the same waveform
crops and the same multi-resolution mel metric, avoiding noisy single-step
training-log comparisons.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.data.kss import DEFAULT_INDEX, KSSDataset
from training.losses.ae_losses import MultiResolutionMelLoss
from training.models.ae import SpeechAutoencoder


def infer_arch_kwargs(ck: dict) -> dict:
    cfg = ck.get("cfg", {}) or {}
    state = ck["ae"]

    stem_in = state["encoder.stem.weight"].shape[1]
    spec_mode = cfg.get("spec_mode")
    if spec_mode is None:
        spec_mode = "concat" if stem_in == 1253 else "mel"

    return dict(
        spec_mode=spec_mode,
        encoder_pad_mode=cfg.get("encoder_pad_mode", cfg.get("pad_mode", "symmetric")),
        decoder_pad_mode=cfg.get("decoder_pad_mode", "causal"),
        enable_encoder_stem_bn=cfg.get(
            "enable_encoder_stem_bn",
            any(k.startswith("encoder.stem_norm.") for k in state),
        ),
        enable_encoder_out_ln=cfg.get(
            "enable_encoder_out_ln",
            any(k.startswith("encoder.out_norm.") for k in state),
        ),
        enable_decoder_stem_bn=cfg.get(
            "enable_decoder_stem_bn",
            any(k.startswith("decoder.stem_norm.") for k in state),
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="train_ae.py checkpoint paths")
    ap.add_argument("--index_path", type=str, default=str(DEFAULT_INDEX))
    ap.add_argument("--crop_seconds", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260506)
    ap.add_argument("--reduction", choices=["mean", "sum"], default="mean")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds = KSSDataset(args.index_path, crop_seconds=args.crop_seconds, sample_rate=44100)
    n = min(args.n, len(ds))
    wavs = torch.stack([ds[i] for i in range(n)], dim=0)
    batches = list(wavs.split(args.batch_size))

    loss_fn = MultiResolutionMelLoss(sample_rate=44100, reduction=args.reduction).to(device).eval()

    print(f"[info] device={device} crop={args.crop_seconds}s n={n} reduction={args.reduction}")
    print(f"{'checkpoint':<44} {'step':>8} {'mel':>10} {'z_std':>9} {'wav_abs':>9}")

    for ckpt_path in args.ckpts:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        ae = SpeechAutoencoder(**infer_arch_kwargs(ck)).to(device)
        ae.load_state_dict(ck["ae"])
        ae.eval()

        vals: list[float] = []
        z_stds: list[float] = []
        wav_abs: list[float] = []
        with torch.no_grad():
            for batch in batches:
                wav = batch.to(device)
                wav_hat, z = ae(wav)
                target_len = wav.shape[1]
                if wav_hat.shape[1] >= target_len:
                    wav_hat = wav_hat[:, :target_len]
                else:
                    wav_hat = torch.nn.functional.pad(wav_hat, (0, target_len - wav_hat.shape[1]))
                vals.append(loss_fn(wav_hat, wav).item())
                z_stds.append(z.std().item())
                wav_abs.append(wav_hat.abs().mean().item())

        name = Path(ckpt_path).as_posix()
        if len(name) > 44:
            name = "..." + name[-41:]
        print(
            f"{name:<44} {ck.get('step', '?'):>8} "
            f"{float(np.mean(vals)):>10.4f} {float(np.mean(z_stds)):>9.3f} "
            f"{float(np.mean(wav_abs)):>9.4f}"
        )

        del ae, ck
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
