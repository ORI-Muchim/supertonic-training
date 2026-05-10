"""Synthesize full-length KSS utterances with a trained train_ae.py ckpt.

Loads encoder+decoder from a single AE checkpoint (paper recipe, no shipped
vocoder). For listen-tests of from-scratch GAN training progress.

Run:
  python -m training.scripts.synth_ae \
      --ckpt training/runs/ae_pilot_30k_r1ttur/ckpt_step00010000.pt \
      --n 8 --out_dir training/runs/ae_pilot_30k_r1ttur/synth_10k
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import torch
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.models.ae import SpeechAutoencoder
from training.data.kss import KSSFullUtteranceDataset, DEFAULT_INDEX


@torch.no_grad()
def synth_one(wav: torch.Tensor, ae: SpeechAutoencoder, device):
    wav = wav.to(device).unsqueeze(0)            # [1, T]
    wav_hat, _z = ae(wav)                        # [1, T_hat]
    return wav_hat.squeeze(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="train_ae.py ckpt (must have 'ae' key in state dict)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--index_path", type=str, default=str(DEFAULT_INDEX))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] device: {device}  out_dir: {out_dir}")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "ae" not in ck:
        raise KeyError(f"ckpt has no 'ae' key (only {list(ck.keys())}). "
                       "synth_ae.py expects train_ae.py format; for "
                       "train_ae_ft.py ckpts use synth_ae_ft.py instead.")
    # Read arch-affecting fields from ckpt's saved cfg.
    saved_cfg = ck.get("cfg", {})
    arch_kwargs = dict(
        spec_mode=saved_cfg.get("spec_mode", "concat"),
        encoder_pad_mode=saved_cfg.get("encoder_pad_mode", saved_cfg.get("pad_mode", "symmetric")),
        decoder_pad_mode=saved_cfg.get("decoder_pad_mode", "causal"),
        enable_encoder_stem_bn=saved_cfg.get("enable_encoder_stem_bn", True),
        enable_encoder_out_ln=saved_cfg.get("enable_encoder_out_ln", True),
        enable_decoder_stem_bn=saved_cfg.get("enable_decoder_stem_bn", True),
    )
    print(f"[info] ckpt cfg: {arch_kwargs}")
    ae = SpeechAutoencoder(**arch_kwargs).to(device)
    ae.load_state_dict(ck["ae"])
    ae.eval()
    step = ck.get("step", "?")
    print(f"[info] AE loaded from {args.ckpt}  (step {step})")

    ds = KSSFullUtteranceDataset(args.index_path)
    print(f"[info] dataset: {len(ds)} utterances")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=args.n, replace=False)

    for idx in indices:
        item = ds[int(idx)]
        wav = item["wav"]
        path_rel = item["path"].replace("/", "_").replace("\\", "_")
        dur = item["duration"]
        wav_hat = synth_one(wav, ae, device)

        T_real = wav.shape[0]; T_fake = wav_hat.shape[0]
        if T_fake >= T_real:
            wav_hat = wav_hat[:T_real]
        else:
            wav_hat = np.pad(wav_hat, (0, T_real - T_fake))

        stem = f"step{step:08d}_idx{int(idx):05d}_{path_rel.replace('.wav','')}"
        sf.write(str(out_dir / f"{stem}_real.wav"), wav.numpy(), 44100)
        sf.write(str(out_dir / f"{stem}_fake.wav"),
                 np.clip(wav_hat, -1, 1), 44100)
        print(f"  [{int(idx):5d}] {path_rel}  dur={dur:.2f}s  saved.")

    print(f"[done] {args.n} utterance pairs in {out_dir}")


if __name__ == "__main__":
    main()
