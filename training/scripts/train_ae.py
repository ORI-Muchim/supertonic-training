"""AE training (Stage 1): GAN-based speech autoencoder.

Paper recipe (arXiv 2503.23108):
  L_G = 45 · L_mel + 1 · L_adv + 0.1 · L_fm
  AdamW, lr=2e-4, batch=128, 1.5M steps (paper: 4×RTX 4090)

KSS-adapted (RTX 3090 × 1, 12.86 h single-speaker Korean):
  default: batch=8, crop=1.0s, lr=2e-4, ~300k steps expected to converge

Run:
  python -m training.scripts.train_ae --smoke          # 500-step smoke test
  python -m training.scripts.train_ae --steps 300000   # real run
"""
from __future__ import annotations
import os, sys, json, argparse, time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.models.ae import SpeechAutoencoder
from training.models.discriminators import AEDiscriminator
from training.losses.ae_losses import (
    MultiResolutionMelLoss, generator_adv_loss,
    discriminator_adv_loss, feature_matching_loss,
)
from training.data.kss import KSSDataset, DEFAULT_INDEX


@dataclass
class AEConfig:
    # data
    index_path: str = str(DEFAULT_INDEX)
    crop_seconds: float = 1.0
    sample_rate: int = 44100
    # model
    pad_mode: str = "causal"
    # loss weights
    w_mel: float = 45.0
    w_adv: float = 1.0
    w_fm:  float = 0.1
    # optim
    lr: float = 2e-4
    beta1: float = 0.8
    beta2: float = 0.99
    weight_decay: float = 0.0
    # schedule
    steps: int = 1_500_000
    batch_size: int = 8
    num_workers: int = 2
    grad_clip: float = 10.0
    # logging
    log_every: int = 50
    sample_every: int = 2_000
    ckpt_every: int = 10_000
    out_dir: str = "training/runs/ae"
    # behavior
    warmup_d_steps: int = 100       # train G only (no D) for first N steps
    resume: str | None = None


def make_loaders(cfg: AEConfig):
    ds = KSSDataset(cfg.index_path, crop_seconds=cfg.crop_seconds, sample_rate=cfg.sample_rate)
    dl = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    return ds, dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="500-step sanity run on tiny subset")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = AEConfig()
    if args.steps is not None:      cfg.steps = args.steps
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.out_dir is not None:    cfg.out_dir = args.out_dir
    if args.resume is not None:     cfg.resume = args.resume
    if args.smoke:
        cfg.steps = 500
        cfg.batch_size = 4
        cfg.log_every = 10
        cfg.sample_every = 100
        cfg.ckpt_every = 10_000
        cfg.out_dir = "training/runs/ae_smoke"

    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device(args.device)
    print(f"[info] device: {device}")
    print(f"[info] out_dir: {out_dir}")
    print(f"[info] cfg: {asdict(cfg)}")

    # --- data ---
    _, dl = make_loaders(cfg)
    data_iter = iter(_inf_loader(dl))

    # --- models ---
    ae = SpeechAutoencoder(pad_mode=cfg.pad_mode).to(device)
    D  = AEDiscriminator().to(device)
    n_g = sum(p.numel() for p in ae.parameters())
    n_d = sum(p.numel() for p in D.parameters())
    print(f"[info] params: generator={n_g/1e6:.2f}M, discriminator={n_d/1e6:.2f}M")

    # --- losses ---
    mel_loss = MultiResolutionMelLoss(sample_rate=cfg.sample_rate).to(device)

    # --- optim ---
    opt_g = torch.optim.AdamW(ae.parameters(),
                              lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    opt_d = torch.optim.AdamW(D.parameters(),
                              lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

    # --- resume ---
    step0 = 0
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=device, weights_only=False)
        ae.load_state_dict(ck["ae"]);        D.load_state_dict(ck["D"])
        opt_g.load_state_dict(ck["opt_g"]);  opt_d.load_state_dict(ck["opt_d"])
        step0 = ck["step"]
        print(f"[info] resumed from step {step0}")

    # --- tensorboard ---
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(out_dir / "tb"))
    except Exception:
        tb = None

    # --- train ---
    ae.train(); D.train()
    t_start = time.time()
    for step in range(step0 + 1, cfg.steps + 1):
        wav = next(data_iter).to(device, non_blocking=True)   # [B, T_crop]
        B, T_in = wav.shape

        # ===== Generator forward =====
        wav_hat, z = ae(wav)
        # align lengths (wav_hat is wav_hat longer due to STFT center padding)
        T_out = wav_hat.shape[1]
        if T_out >= T_in:
            wav_hat = wav_hat[:, :T_in]
        else:
            pad = T_in - T_out
            wav_hat = torch.nn.functional.pad(wav_hat, (0, pad))

        # ===== Discriminator step =====
        if step > cfg.warmup_d_steps:
            real_logits, _ = D(wav)
            fake_logits, _ = D(wav_hat.detach())
            L_D = discriminator_adv_loss(real_logits, fake_logits)
            opt_d.zero_grad(set_to_none=True)
            L_D.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.grad_clip)
            opt_d.step()
        else:
            L_D = torch.tensor(0.0, device=device)

        # ===== Generator step =====
        L_mel = mel_loss(wav_hat, wav)
        if step > cfg.warmup_d_steps:
            fake_logits_g, fake_feats = D(wav_hat)
            _,             real_feats = D(wav)
            L_adv = generator_adv_loss(fake_logits_g)
            L_fm  = feature_matching_loss(real_feats, fake_feats)
        else:
            L_adv = torch.tensor(0.0, device=device)
            L_fm  = torch.tensor(0.0, device=device)

        L_G = cfg.w_mel * L_mel + cfg.w_adv * L_adv + cfg.w_fm * L_fm
        opt_g.zero_grad(set_to_none=True)
        L_G.backward()
        g_grad_norm = torch.nn.utils.clip_grad_norm_(ae.parameters(), cfg.grad_clip)
        opt_g.step()

        # ===== log =====
        if step % cfg.log_every == 0:
            elapsed = time.time() - t_start
            sps = (step - step0) / max(elapsed, 1e-9)
            msg = (
                f"[step {step}/{cfg.steps}]  "
                f"L_G={L_G.item():.4f}  mel={L_mel.item():.4f}  adv={L_adv.item():.4f}  "
                f"fm={L_fm.item():.4f}  L_D={L_D.item():.4f}  "
                f"gn={g_grad_norm:.2f}  |  {sps:.2f} step/s"
            )
            print(msg, flush=True)
            if tb is not None:
                tb.add_scalar("G/total", L_G.item(), step)
                tb.add_scalar("G/mel",   L_mel.item(), step)
                tb.add_scalar("G/adv",   L_adv.item(), step)
                tb.add_scalar("G/fm",    L_fm.item(), step)
                tb.add_scalar("D/total", L_D.item(), step)
                tb.add_scalar("grad/G_norm", g_grad_norm.item(), step)
                tb.add_scalar("sys/step_per_sec", sps, step)

        # ===== save audio samples =====
        if step % cfg.sample_every == 0:
            import soundfile as sf
            samp_dir = out_dir / "samples"; samp_dir.mkdir(exist_ok=True)
            for i in range(min(2, B)):
                sf.write(str(samp_dir / f"step{step:08d}_{i}_real.wav"),
                         wav[i].detach().cpu().numpy(), cfg.sample_rate)
                sf.write(str(samp_dir / f"step{step:08d}_{i}_fake.wav"),
                         wav_hat[i].detach().cpu().numpy().clip(-1, 1), cfg.sample_rate)

        # ===== checkpoint =====
        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            ck_path = out_dir / f"ckpt_step{step:08d}.pt"
            torch.save({
                "step": step,
                "ae": ae.state_dict(), "D": D.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "cfg": asdict(cfg),
            }, ck_path)
            print(f"[ckpt] saved {ck_path}")

    print(f"[done] total time: {(time.time()-t_start)/60:.1f} min")


def _inf_loader(dl):
    while True:
        for b in dl:
            yield b


if __name__ == "__main__":
    main()
