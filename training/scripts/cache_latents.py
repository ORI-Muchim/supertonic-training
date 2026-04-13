"""Cache AE latents for TTL (Stage 2) training.

After Stage-1 AE training, run the frozen encoder over every KSS utterance and save
each latent z_1 to disk. Also compute the per-dim mean/std for normalization
(feeding z into TTL in normalized space, matching the normalizer.scale trick).

Output structure:
    training/runs/ae_v1/latents/
        manifest.json               # [{idx, path, latent_path, T, duration, text_raw, text_norm}, ...]
        stats.pt                    # {"mean": [24], "std": [24], "n_samples": ...}
        latents/
            0000000.pt              # Tensor [24, T], float16 for compactness
            0000001.pt
            ...

Usage:
    python -m training.scripts.cache_latents \
        --ckpt training/runs/ae_v1/ckpt_step00300000.pt \
        --out_dir training/runs/ae_v1/cache
"""
from __future__ import annotations
import os, sys, json, argparse, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.models.ae import SpeechAutoencoder
from training.data.kss import KSSFullUtteranceDataset, DEFAULT_INDEX


def collate_single(batch):
    """Single-utterance batch (list of 1 dict). Return without padding."""
    assert len(batch) == 1
    return batch[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="AE checkpoint from train_ae.py (with `ae` key in state dict)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--index_path", type=str, default=str(DEFAULT_INDEX))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true", help="Save latents in float16 to save disk")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    latents_dir = out_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # --- load AE ---
    # Accepts two ckpt layouts:
    #   (a) train_ae.py saves {"ae": full_SpeechAutoencoder_state_dict, ...}
    #   (b) train_ae_ft.py saves {"encoder": AEEncoder_state_dict, ...}  (decoder was frozen/shipped)
    print(f"[info] loading AE from {args.ckpt}")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    ae = SpeechAutoencoder().to(device)
    if "ae" in ck:
        ae.load_state_dict(ck["ae"])
    elif "encoder" in ck:
        ae.encoder.load_state_dict(ck["encoder"])
        print("[info] loaded encoder-only ckpt (from train_ae_ft.py); decoder left at init "
              "- this is fine because we only use the encoder here.")
    else:
        raise KeyError(f"ckpt has no 'ae' or 'encoder' key: {list(ck.keys())}")
    ae.eval()
    encoder = ae.encoder
    spec = ae.spec

    # --- dataset ---
    ds = KSSFullUtteranceDataset(args.index_path)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_single)
    print(f"[info] processing {len(ds)} utterances")

    # --- running stats (Welford-ish, keep mean and sum-of-squares per-dim) ---
    running_mean = torch.zeros(24, device=device, dtype=torch.float64)
    running_m2   = torch.zeros(24, device=device, dtype=torch.float64)
    n_frames_total = 0

    manifest: list[dict] = []
    t0 = time.time()
    dtype = torch.float16 if args.fp16 else torch.float32

    with torch.no_grad():
        for idx, item in enumerate(dl):
            wav = item["wav"].to(device).unsqueeze(0)   # [1, T_samples]
            feats = spec(wav)                            # [1, 1253, T_frames]
            z = encoder(feats)                           # [1, 24, T_frames]
            z = z.squeeze(0)                             # [24, T]

            # update running stats (on f32 to preserve precision)
            z64 = z.to(torch.float64)
            Ti = z64.shape[-1]
            delta = z64 - running_mean.unsqueeze(-1)
            running_mean = running_mean + delta.sum(dim=-1) / max(n_frames_total + Ti, 1)
            running_m2   = running_m2   + (delta * (z64 - running_mean.unsqueeze(-1))).sum(dim=-1)
            n_frames_total += Ti

            # save
            out_path = latents_dir / f"{idx:07d}.pt"
            torch.save(z.to(dtype).cpu(), out_path)
            manifest.append({
                "idx": idx,
                "path": item["path"],
                "latent_path": str(out_path.relative_to(out_dir)),
                "T_frames": int(Ti),
                "duration": item["duration"],
                "text_raw": item["text_raw"],
                "text_norm": item["text_norm"],
            })

            if (idx + 1) % 500 == 0:
                sps = (idx + 1) / max(time.time() - t0, 1e-9)
                print(f"  [{idx+1}/{len(ds)}]  {sps:.1f} utt/s  "
                      f"total_frames={n_frames_total:,}", flush=True)

    # --- finalize stats ---
    stats = {
        "mean": running_mean.float().cpu(),
        "std":  (running_m2 / max(n_frames_total - 1, 1)).sqrt().float().cpu(),
        "n_frames": n_frames_total,
        "n_utterances": len(ds),
    }
    torch.save(stats, out_dir / "stats.pt")
    print(f"[stats] mean per-dim range: [{stats['mean'].min():.3f}, {stats['mean'].max():.3f}]")
    print(f"[stats] std  per-dim range: [{stats['std'].min():.3f}, {stats['std'].max():.3f}]")

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)

    total_time = time.time() - t0
    print(f"[done] {len(manifest)} latents saved. total_time={total_time/60:.1f} min")
    print(f"       cached under: {out_dir}")


if __name__ == "__main__":
    main()
