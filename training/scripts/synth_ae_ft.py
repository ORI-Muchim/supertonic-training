"""Synthesize full-length KSS utterances with a trained AE-FT encoder.

Loads:
  - encoder ckpt (training/runs/ae_ft_optA/ckpt_step00060000.pt)
  - shipped vocoder.onnx (frozen, weights via analysis/torch_vocoder)

Pipeline (matches train_ae_ft.py forward):
  wav -> spec -> encoder -> instance_norm -> z_norm * scale -> chunk -> vocoder -> wav_hat

Saves real/fake pairs to <out_dir>/*.wav.

Run:
  python -m training.scripts.synth_ae_ft \
      --ckpt training/runs/ae_ft_optA/ckpt_step00060000.pt \
      --n 8 --out_dir training/runs/ae_ft_optA/synth_60k
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from training.data.kss import KSSFullUtteranceDataset
from training.data.spectrogram import SpecProcessor
from training.models.ae_encoder import AEEncoder
from training.scripts.train_ae_ft import chunk_compress

from torch_vocoder import Vocoder, load_vocoder_weights  # type: ignore


@torch.no_grad()
def synth_one(wav: torch.Tensor, encoder, spec, vocoder,
              latent_mean, latent_std, normalizer_scale, device):
    wav = wav.to(device).unsqueeze(0)            # [1, T]
    feats = spec(wav)                            # [1, 1253, T_f]
    z_raw = encoder(feats)                       # [1, 24, T_f]

    # SAME instance norm as training
    mu  = z_raw.mean(dim=-1, keepdim=True)
    sig = z_raw.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
    z_norm = (z_raw - mu) / sig

    z_ttl = chunk_compress(z_norm * normalizer_scale, kc=6)
    wav_hat = vocoder(z_ttl)                     # [1, T_hat]
    return wav_hat.squeeze(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--vocoder_onnx", type=str,
                    default=str(ROOT / "assets" / "onnx" / "vocoder.onnx"))
    ap.add_argument("--index_path", type=str,
                    default=str(ROOT / "training" / "data" / "kss_index.json"))
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n", type=int, default=8, help="number of utterances")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    # AE config (must match training)
    ap.add_argument("--ldim", type=int, default=24)
    ap.add_argument("--hdim", type=int, default=512)
    ap.add_argument("--intermediate_dim", type=int, default=2048)
    ap.add_argument("--num_layers", type=int, default=10)
    ap.add_argument("--ksz", type=int, default=7)
    ap.add_argument("--pad_mode", type=str, default="causal")
    ap.add_argument("--ttl_normalizer_scale", type=float, default=0.25)
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] device: {device}  out_dir: {out_dir}")

    # spec
    spec = SpecProcessor(sample_rate=44100, n_fft=2048, win_length=2048,
                         hop_length=512, n_mels=228).to(device)

    # encoder
    encoder = AEEncoder(
        idim=spec.feature_dim, hdim=args.hdim, odim=args.ldim,
        ksz_init=args.ksz, ksz=args.ksz, num_layers=args.num_layers,
        intermediate_dim=args.intermediate_dim, pad_mode=args.pad_mode,
    ).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(ck["encoder"])
    encoder.eval()
    print(f"[info] encoder loaded from {args.ckpt}  (step {ck.get('step', '?')})")

    # vocoder
    vocoder = Vocoder(ldim=args.ldim, chunk_compress_factor=6, hdim=args.hdim,
                      intermediate=args.intermediate_dim, num_layers=args.num_layers,
                      ttl_normalizer_scale=args.ttl_normalizer_scale).to(device)
    load_vocoder_weights(vocoder, args.vocoder_onnx)
    vocoder.eval()
    latent_mean = vocoder.latent_mean.detach()
    latent_std  = vocoder.latent_std.detach()
    normalizer_scale = float(vocoder.normalizer_scale.item())
    print(f"[info] vocoder loaded from {args.vocoder_onnx}")

    # dataset (full utterances)
    ds = KSSFullUtteranceDataset(args.index_path)
    print(f"[info] dataset: {len(ds)} utterances")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=args.n, replace=False)

    for idx in indices:
        item = ds[int(idx)]
        wav = item["wav"]                 # [T_samples], float32
        path_rel = item["path"].replace("/", "_").replace("\\", "_")
        dur = item["duration"]
        wav_hat = synth_one(wav, encoder, spec, vocoder,
                            latent_mean, latent_std, normalizer_scale, device)

        # length-align: trim/pad fake to match real
        T_real = wav.shape[0]; T_fake = wav_hat.shape[0]
        if T_fake >= T_real:
            wav_hat = wav_hat[:T_real]
        else:
            wav_hat = np.pad(wav_hat, (0, T_real - T_fake))

        stem = f"idx{int(idx):05d}_{path_rel.replace('.wav','')}"
        sf.write(str(out_dir / f"{stem}_real.wav"), wav.numpy(), 44100)
        sf.write(str(out_dir / f"{stem}_fake.wav"), wav_hat, 44100)
        print(f"  [{int(idx):5d}] {path_rel}  dur={dur:.2f}s  saved.")

    print(f"[done] {args.n} utterance pairs in {out_dir}")


if __name__ == "__main__":
    main()
