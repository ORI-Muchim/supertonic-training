"""Stage 1 fine-tune: train only AEEncoder, with shipped vocoder.onnx weights
loaded into a FROZEN Vocoder decoder.

Why not train from scratch?
  - Paper uses 1.5M steps on 4 GPUs (~100 GPU-days).
  - On a single 3090 at batch 16 (~1.7 step/s), that's > month.
  - GAN dynamics are also fragile on small batch (discriminator wins, gradient
    explodes through adv path, see earlier diagnosis).

By freezing the shipped decoder (which is already paper quality), we reduce
the problem to: train an encoder that produces AE latents the decoder can
decode back to intelligible waveform. This is tractable in hours.

Loss:
  L = 45 * L_wav + λ_lat * L_latent
  L_wav     : multi-res log-mel L1 (same as paper's reconstruction term)
  L_latent  : moment-matching penalty keeping z_ae ~ N(shipped_mean, shipped_std²)
              so the frozen decoder's pre-baked de-normalization stays on-manifold.

No discriminator. No feature matching. Pure reconstruction with a distribution
constraint — much more stable than adversarial training at small batch.

Run:
  python -m training.scripts.train_ae_ft --smoke
  python -m training.scripts.train_ae_ft --steps 60000 --out_dir training/runs/ae_ft
"""
from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from training.data.kss import KSSDataset
from training.data.spectrogram import SpecProcessor
from training.models.ae_encoder import AEEncoder
from training.losses.ae_losses import MultiResolutionMelLoss

from torch_vocoder import Vocoder, load_vocoder_weights  # type: ignore


@dataclass
class AEFTConfig:
    # data
    index_path: str = str(ROOT / "training" / "data" / "kss_index.json")
    crop_seconds: float = 1.0
    sample_rate: int = 44100
    # shipped vocoder
    vocoder_onnx: str = str(ROOT / "assets" / "onnx" / "vocoder.onnx")
    # model
    ldim: int = 24
    hdim: int = 512
    intermediate_dim: int = 2048
    num_layers: int = 10
    ksz: int = 7
    pad_mode: str = "causal"
    # ttl scale — applied between normalized AE latent and the frozen decoder
    ttl_normalizer_scale: float = 0.25
    # loss
    w_wav: float = 45.0
    w_latent: float = 0.0   # 0 because instance-norm makes L_latent ~0 by construction
    instance_norm_z: bool = True   # hard per-(sample,channel) normalize at encoder output
    # optim
    lr: float = 2e-4
    beta1: float = 0.8
    beta2: float = 0.99
    weight_decay: float = 0.0
    grad_clip: float = 10.0
    # schedule
    steps: int = 60_000
    batch_size: int = 16
    num_workers: int = 0      # Windows-safe default
    # logging
    log_every: int = 50
    sample_every: int = 2_000
    ckpt_every: int = 10_000
    out_dir: str = "training/runs/ae_ft"
    resume: str | None = None


def chunk_compress(z: torch.Tensor, kc: int) -> torch.Tensor:
    """[B, C, T] -> [B, C*kc, T/kc] with sub-pixel layout matching the decoder's
    inverse un-chunk. If T isn't divisible by kc, right-pad with replicate."""
    B, C, T = z.shape
    if T % kc != 0:
        pad = kc - (T % kc)
        z = F.pad(z, (0, pad), mode="replicate")
        T = z.shape[-1]
    z = z.reshape(B, C, T // kc, kc).permute(0, 1, 3, 2).contiguous()  # [B, C, kc, T/kc]
    z = z.reshape(B, C * kc, T // kc)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="500-step sanity run")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = AEFTConfig()
    if args.steps is not None:      cfg.steps = args.steps
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.out_dir is not None:    cfg.out_dir = args.out_dir
    if args.resume is not None:     cfg.resume = args.resume
    if args.num_workers is not None: cfg.num_workers = args.num_workers
    if args.smoke:
        cfg.steps = 500
        cfg.batch_size = 4
        cfg.log_every = 10
        cfg.sample_every = 100
        cfg.ckpt_every = 1_000_000
        cfg.out_dir = "training/runs/ae_ft_smoke"

    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    device = torch.device(args.device)
    print(f"[info] device: {device}")
    print(f"[info] out_dir: {cfg.out_dir}")

    # --- spectrogram + encoder (trainable) ---
    spec = SpecProcessor(sample_rate=cfg.sample_rate, n_fft=2048, win_length=2048,
                         hop_length=512, n_mels=228).to(device)
    idim = spec.feature_dim
    encoder = AEEncoder(
        idim=idim, hdim=cfg.hdim, odim=cfg.ldim, ksz_init=cfg.ksz, ksz=cfg.ksz,
        num_layers=cfg.num_layers, intermediate_dim=cfg.intermediate_dim,
        pad_mode=cfg.pad_mode,
    ).to(device)
    enc_n = sum(p.numel() for p in encoder.parameters())
    print(f"[info] AEEncoder params: {enc_n/1e6:.2f}M (trainable)")

    # --- shipped vocoder (frozen) ---
    vocoder = Vocoder(ldim=cfg.ldim, chunk_compress_factor=6, hdim=cfg.hdim,
                      intermediate=cfg.intermediate_dim, num_layers=cfg.num_layers,
                      ttl_normalizer_scale=cfg.ttl_normalizer_scale).to(device)
    load_vocoder_weights(vocoder, cfg.vocoder_onnx)
    for p in vocoder.parameters():
        p.requires_grad_(False)
    vocoder.eval()  # CRITICAL: keeps BatchNorm running stats frozen
    voc_n = sum(p.numel() for p in vocoder.parameters())
    print(f"[info] Vocoder params: {voc_n/1e6:.2f}M (frozen, loaded from {cfg.vocoder_onnx})")

    # Shipped decoder contract:
    #   decoder expects post-normalized AE latent:  z_norm ~ N(0, 1) per channel
    #   it recovers raw via: x = z_norm * latent_std + latent_mean (inside vocoder.forward)
    # So our encoder must produce z_ae with distribution:
    #     (z_ae - latent_mean) / latent_std ~ N(0, 1)
    # That's what the moment-matching loss enforces.
    latent_mean = vocoder.latent_mean.detach()   # [1, 24, 1]
    latent_std  = vocoder.latent_std.detach()    # [1, 24, 1]
    normalizer_scale = float(vocoder.normalizer_scale.item())
    print(f"[info] shipped latent_mean range: [{latent_mean.min().item():.3f}, "
          f"{latent_mean.max().item():.3f}]  std range: [{latent_std.min().item():.3f}, "
          f"{latent_std.max().item():.3f}]  normalizer_scale: {normalizer_scale:.3f}")

    # --- losses ---
    wav_loss = MultiResolutionMelLoss(sample_rate=cfg.sample_rate).to(device)

    # --- data ---
    ds = KSSDataset(cfg.index_path, crop_seconds=cfg.crop_seconds, sample_rate=cfg.sample_rate)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, drop_last=True, pin_memory=True,
                    persistent_workers=cfg.num_workers > 0)
    def inf():
        while True:
            for b in dl: yield b
    data_iter = iter(inf())
    print(f"[info] dataset: {len(ds)} utterances, crop {cfg.crop_seconds}s")

    # --- optim ---
    opt = torch.optim.AdamW(encoder.parameters(),
                            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                            weight_decay=cfg.weight_decay)

    step0 = 0
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(ck["encoder"])
        opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        print(f"[info] resumed from step {step0}")

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(out_dir / "tb"))
    except Exception:
        tb = None

    encoder.train()   # decoder stays in eval; enforced above
    t_start = time.time()

    for step in range(step0 + 1, cfg.steps + 1):
        wav = next(data_iter).to(device, non_blocking=True)
        B, T_wav = wav.shape

        # === forward ===
        feats = spec(wav)                       # [B, 1253, T_frames]
        z_raw = encoder(feats)                  # [B, 24, T_frames]

        # Hard instance-normalize per (sample, channel) over time so z_norm has
        # per-(B,c) mean=0/std=1 by construction. Pooled across (B,T) it is also
        # (0, 1), so frozen vocoder's internal `× latent_std + latent_mean`
        # de-normalize lands the signal in shipped's raw latent distribution.
        # This removes the moment-matching gradient war that destabilized w_latent>0.
        if cfg.instance_norm_z:
            mu  = z_raw.mean(dim=-1, keepdim=True)
            sig = z_raw.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
            z_norm = (z_raw - mu) / sig
            z_ae = z_norm * latent_std + latent_mean    # for ckpt parity / downstream
        else:
            z_ae = z_raw
            z_norm = (z_ae - latent_mean) / latent_std    # legacy path

        z_ttl = chunk_compress(z_norm * normalizer_scale, kc=6)  # [B, 144, T/6]

        wav_hat = vocoder(z_ttl)                 # [B, T_hat]

        # Align wav lengths (vocoder pads due to stem)
        T_hat = wav_hat.shape[1]
        if T_hat >= T_wav:
            wav_hat = wav_hat[:, :T_wav]
        else:
            wav_hat = F.pad(wav_hat, (0, T_wav - T_hat))

        # === losses ===
        L_wav = wav_loss(wav_hat, wav)

        # Moment matching on z_norm across (B, T). Per-channel, squared-sum so
        # a single exploding channel can't be diluted by 23 well-behaved ones.
        z_flat = z_norm.transpose(1, 2).reshape(-1, cfg.ldim)   # [B*T, 24]
        z_mean = z_flat.mean(dim=0)              # [24]
        z_std  = z_flat.std(dim=0).clamp_min(1e-6)
        L_latent = (z_mean ** 2).sum() + ((z_std - 1.0) ** 2).sum()

        L = cfg.w_wav * L_wav + cfg.w_latent * L_latent

        # === backward ===
        opt.zero_grad(set_to_none=True)
        L.backward()
        gn = torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.grad_clip)
        opt.step()

        # === log ===
        if step % cfg.log_every == 0:
            elapsed = time.time() - t_start
            sps = (step - step0) / max(elapsed, 1e-9)
            z_sigma_max = z_std.max().item()
            z_sigma_min = z_std.min().item()
            z_mu_max = z_mean.abs().max().item()
            print(
                f"[step {step}/{cfg.steps}]  "
                f"L={L.item():.4f}  wav={L_wav.item():.4f}  "
                f"lat={L_latent.item():.4f}  "
                f"zmu_mean={z_mean.abs().mean().item():.3f}  zmu_max={z_mu_max:.3f}  "
                f"zsig_mean={z_std.mean().item():.3f}  zsig=[{z_sigma_min:.3f},{z_sigma_max:.3f}]  "
                f"gn={gn:.2f}  |  {sps:.2f} step/s",
                flush=True,
            )
            if tb is not None:
                tb.add_scalar("L/total",  L.item(), step)
                tb.add_scalar("L/wav",    L_wav.item(), step)
                tb.add_scalar("L/latent", L_latent.item(), step)
                tb.add_scalar("z/mu_abs_mean", z_mean.abs().mean().item(), step)
                tb.add_scalar("z/mu_abs_max",  z_mu_max, step)
                tb.add_scalar("z/sigma_mean",  z_std.mean().item(), step)
                tb.add_scalar("z/sigma_max",   z_sigma_max, step)
                tb.add_scalar("z/sigma_min",   z_sigma_min, step)
                tb.add_scalar("grad/norm", gn.item(), step)
                tb.add_scalar("sys/sps",   sps, step)

        # === sample audio ===
        if step % cfg.sample_every == 0:
            import soundfile as sf
            samp = out_dir / "samples"; samp.mkdir(exist_ok=True)
            for i in range(min(2, B)):
                sf.write(str(samp / f"step{step:08d}_{i}_real.wav"),
                         wav[i].detach().cpu().numpy(), cfg.sample_rate)
                sf.write(str(samp / f"step{step:08d}_{i}_fake.wav"),
                         wav_hat[i].detach().cpu().numpy(), cfg.sample_rate)

        # === ckpt ===
        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            ck_path = out_dir / f"ckpt_step{step:08d}.pt"
            torch.save({
                "step": step,
                "encoder": encoder.state_dict(),
                "opt": opt.state_dict(),
                "cfg": asdict(cfg),
            }, ck_path)
            print(f"[ckpt] {ck_path}")

    print(f"[done] total time: {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
