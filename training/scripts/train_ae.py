"""AE training (Stage 1): GAN-based speech autoencoder.

Paper recipe (arXiv 2503.23108):
  L_G = 45 · L_mel + 1 · L_adv + 0.1 · L_fm
  AdamW, lr=2e-4, batch=128, 1.5M steps (paper: 4×RTX 4090)

KSS-adapted (RTX 3090 × 1, 12.86 h single-speaker Korean):
  default: batch=16, crop=1.0s, lr=2e-4, ~300k steps expected to converge

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
    """Paper-faithful defaults (SupertonicTTS arXiv 2503.23108, App. B.1).

    Paper: batch_size=128, lr=2e-4, 1.5M iterations on 4×RTX 4090.
    Paper separately specifies 0.19s random crops for adversarial training.
    On a single 3090, use longer input crops for reconstruction context but
    keep the discriminator crop at the paper value.
    """
    # data
    index_path: str = str(DEFAULT_INDEX)
    crop_seconds: float = 1.0       # encoder/reconstruction input crop length
    adv_crop_seconds: float = 0.19  # paper Sec. 3.2: real/generated crop for adversarial training
    sample_rate: int = 44100
    # model (paper: encoder NOT causal, decoder IS causal — Sec 3.1.1, A.1.2)
    encoder_pad_mode: str = "symmetric"
    decoder_pad_mode: str = "causal"
    spec_mode: str = "mel"          # paper Sec 3.1 / B.1: log-mel only (228-dim)
    # loss weights (paper Sec 4.1 / B.1)
    w_mel: float = 45.0
    w_adv: float = 1.0
    w_fm:  float = 0.1
    # optim (paper)
    lr: float = 2e-4                # G learning rate
    lr_d: float = 2e-4              # D learning rate (paper does NOT use TTUR; same as lr)
    beta1: float = 0.8
    beta2: float = 0.99
    weight_decay: float = 0.0
    # schedule (paper: 1.5M iters at batch 128 on 4 GPUs)
    steps: int = 1_500_000
    batch_size: int = 16            # 1×3090 conservative default; tune via --batch_size
    num_workers: int = 2
    grad_clip: float | None = None  # paper does not specify clipping; set >0 to enable
    # logging
    log_every: int = 50
    sample_every: int = 2_000
    ckpt_every: int = 10_000
    out_dir: str = "training/runs/ae"
    # behavior — paper does NOT specify warmup; default off to stay faithful.
    warmup_d_steps: int = 0
    # GAN stabilization knobs (NOT in paper). Default off; enable only as ablation.
    r1_gamma: float = 0.0
    r1_every: int = 16
    resume: str | None = None
    # Architecture ablations (defaults = paper-faithful; toggle off to reproduce broken-arch piece-by-piece)
    enable_encoder_stem_bn: bool = True   # paper A.1.1 stem→BN
    enable_encoder_out_ln:  bool = True   # paper A.1.1 final linear→LN on 24-dim
    enable_decoder_stem_bn: bool = True   # paper A.1.2 stem→BN
    lrecon_reduction: str = "mean"        # paper does not state sum vs mean over resolutions


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
    ap.add_argument("--crop_seconds", type=float, default=None,
                    help="input crop length for encoder/reconstruction training")
    ap.add_argument("--adv_crop_seconds", type=float, default=None,
                    help="aligned real/fake crop length used only for D/adv/FM losses")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--warmup_d_steps", type=int, default=None,
                    help="train G only (no D) for first N steps before adversarial loss kicks in")
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--lr_d", type=float, default=None, help="D learning rate (TTUR)")
    ap.add_argument("--r1_gamma", type=float, default=None, help="R1 gradient penalty weight (0 disables)")
    ap.add_argument("--r1_every", type=int, default=None, help="apply R1 every N D-steps")
    ap.add_argument("--grad_clip", type=float, default=None, help="grad-norm clip; default None/off because the paper does not specify clipping")
    ap.add_argument("--ckpt_every", type=int, default=None, help="checkpoint save interval (default 10000)")
    ap.add_argument("--log_every",  type=int, default=None, help="log line interval (default 50)")
    # Architecture ablation flags (each one OFF reverts that paper-faithful fix)
    ap.add_argument("--no_encoder_stem_bn", action="store_true", help="ablation: drop encoder stem BN")
    ap.add_argument("--no_encoder_out_ln",  action="store_true", help="ablation: drop encoder final LayerNorm on 24-dim latent")
    ap.add_argument("--no_decoder_stem_bn", action="store_true", help="ablation: drop decoder stem BN")
    ap.add_argument("--encoder_pad_mode", type=str, default=None, choices=["symmetric", "causal"],
                    help="ablation: override encoder pad mode (paper=symmetric)")
    ap.add_argument("--lrecon_reduction", type=str, default=None, choices=["mean", "sum"],
                    help="multi-resolution mel reduction; paper does not specify sum vs mean")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = AEConfig()
    if args.steps is not None:          cfg.steps = args.steps
    if args.batch_size is not None:     cfg.batch_size = args.batch_size
    if args.crop_seconds is not None:   cfg.crop_seconds = args.crop_seconds
    if args.adv_crop_seconds is not None: cfg.adv_crop_seconds = args.adv_crop_seconds
    if args.out_dir is not None:        cfg.out_dir = args.out_dir
    if args.resume is not None:         cfg.resume = args.resume
    if args.warmup_d_steps is not None: cfg.warmup_d_steps = args.warmup_d_steps
    if args.num_workers is not None:    cfg.num_workers = args.num_workers
    if args.lr_d is not None:           cfg.lr_d = args.lr_d
    if args.r1_gamma is not None:       cfg.r1_gamma = args.r1_gamma
    if args.r1_every is not None:       cfg.r1_every = args.r1_every
    if args.grad_clip is not None:      cfg.grad_clip = args.grad_clip
    if args.ckpt_every is not None:     cfg.ckpt_every = args.ckpt_every
    if args.log_every is not None:      cfg.log_every = args.log_every
    if args.no_encoder_stem_bn:         cfg.enable_encoder_stem_bn = False
    if args.no_encoder_out_ln:          cfg.enable_encoder_out_ln = False
    if args.no_decoder_stem_bn:         cfg.enable_decoder_stem_bn = False
    if args.encoder_pad_mode is not None: cfg.encoder_pad_mode = args.encoder_pad_mode
    if args.lrecon_reduction is not None: cfg.lrecon_reduction = args.lrecon_reduction
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
    ae = SpeechAutoencoder(
        encoder_pad_mode=cfg.encoder_pad_mode,
        decoder_pad_mode=cfg.decoder_pad_mode,
        spec_mode=cfg.spec_mode,
        enable_encoder_stem_bn=cfg.enable_encoder_stem_bn,
        enable_encoder_out_ln=cfg.enable_encoder_out_ln,
        enable_decoder_stem_bn=cfg.enable_decoder_stem_bn,
    ).to(device)
    D  = AEDiscriminator().to(device)
    print(f"[info] spec_mode={cfg.spec_mode}  encoder.idim={ae.spec.feature_dim}")
    n_g = sum(p.numel() for p in ae.parameters())
    n_d = sum(p.numel() for p in D.parameters())
    print(f"[info] params: generator={n_g/1e6:.2f}M, discriminator={n_d/1e6:.2f}M")

    # --- losses ---
    mel_loss = MultiResolutionMelLoss(sample_rate=cfg.sample_rate, reduction=cfg.lrecon_reduction).to(device)

    # --- optim ---
    opt_g = torch.optim.AdamW(ae.parameters(),
                              lr=cfg.lr,   betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    opt_d = torch.optim.AdamW(D.parameters(),
                              lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    print(f"[info] lr_g={cfg.lr}  lr_d={cfg.lr_d}  (TTUR ratio = {cfg.lr/cfg.lr_d:.2f})")
    print(f"[info] R1: gamma={cfg.r1_gamma}  every={cfg.r1_every} D-steps "
          f"({'on' if cfg.r1_gamma > 0 else 'OFF'})")

    # --- resume ---
    step0 = 0
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=device, weights_only=False)
        ae.load_state_dict(ck["ae"]);        D.load_state_dict(ck["D"])
        opt_g.load_state_dict(ck["opt_g"]);  opt_d.load_state_dict(ck["opt_d"])
        step0 = ck["step"]
        # If user changed lr/lr_d on resume, override (state_dict carries old values).
        for pg in opt_g.param_groups: pg["lr"] = cfg.lr
        for pg in opt_d.param_groups: pg["lr"] = cfg.lr_d
        print(f"[info] resumed from step {step0}, lr_g={cfg.lr} lr_d={cfg.lr_d} (overridden)")

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

        # Paper specifies 0.19s random crops of real/generated audio for the
        # adversarial path. Reconstruction loss stays on the full input crop.
        adv_crop_samples = int(round(cfg.adv_crop_seconds * cfg.sample_rate))
        wav_adv, wav_hat_adv = _aligned_random_crop_pair(wav, wav_hat, adv_crop_samples)

        # ===== Discriminator step =====
        L_R1 = torch.tensor(0.0, device=device)
        if step > cfg.warmup_d_steps:
            # R1 path: real input gets gradient enabled so we can backprop through D(real)
            apply_r1 = cfg.r1_gamma > 0 and (step % cfg.r1_every == 0)
            if apply_r1:
                wav_r1 = wav_adv.detach().requires_grad_(True)
                real_logits, _ = D(wav_r1)
                # Sum logits over all heads, then dL/dx -> gradient norm penalty.
                real_logit_sum = sum(lg.sum() for lg in real_logits)
                grad_real = torch.autograd.grad(
                    outputs=real_logit_sum, inputs=wav_r1,
                    create_graph=True, retain_graph=True,
                )[0]
                # Per-sample mean of squared gradient elements (architecture-independent
                # — sum-version explodes with T_wav=44100). Then mean over batch.
                # Multiplied by r1_every for lazy regularization.
                L_R1 = grad_real.pow(2).reshape(B, -1).mean(dim=1).mean()
                L_R1 = (cfg.r1_gamma * 0.5 * cfg.r1_every) * L_R1
            else:
                real_logits, _ = D(wav_adv)
            fake_logits, _ = D(wav_hat_adv.detach())
            L_D_adv = discriminator_adv_loss(real_logits, fake_logits)
            L_D = L_D_adv + L_R1
            opt_d.zero_grad(set_to_none=True)
            L_D.backward()
            _maybe_clip_grad_norm_(D.parameters(), cfg.grad_clip)
            opt_d.step()
        else:
            L_D = torch.tensor(0.0, device=device)

        # ===== Generator step =====
        L_mel = mel_loss(wav_hat, wav)
        if step > cfg.warmup_d_steps:
            fake_logits_g, fake_feats = D(wav_hat_adv)
            _,             real_feats = D(wav_adv)
            L_adv = generator_adv_loss(fake_logits_g)
            L_fm  = feature_matching_loss(real_feats, fake_feats)
        else:
            L_adv = torch.tensor(0.0, device=device)
            L_fm  = torch.tensor(0.0, device=device)

        L_G = cfg.w_mel * L_mel + cfg.w_adv * L_adv + cfg.w_fm * L_fm
        opt_g.zero_grad(set_to_none=True)
        L_G.backward()
        g_grad_norm = _maybe_clip_grad_norm_(ae.parameters(), cfg.grad_clip)
        opt_g.step()

        # ===== log =====
        if step % cfg.log_every == 0:
            elapsed = time.time() - t_start
            sps = (step - step0) / max(elapsed, 1e-9)
            msg = (
                f"[step {step}/{cfg.steps}]  "
                f"L_G={L_G.item():.4f}  mel={L_mel.item():.4f}  adv={L_adv.item():.4f}  "
                f"fm={L_fm.item():.4f}  L_D={L_D.item():.4f}  R1={L_R1.item():.4f}  "
                f"gn={g_grad_norm:.2f}  |  {sps:.2f} step/s"
            )
            print(msg, flush=True)
            if tb is not None:
                tb.add_scalar("G/total", L_G.item(), step)
                tb.add_scalar("G/mel",   L_mel.item(), step)
                tb.add_scalar("G/adv",   L_adv.item(), step)
                tb.add_scalar("G/fm",    L_fm.item(), step)
                tb.add_scalar("D/total", L_D.item(), step)
                tb.add_scalar("D/R1",    L_R1.item(), step)
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


def _aligned_random_crop_pair(
    wav: torch.Tensor,
    wav_hat: torch.Tensor,
    crop_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop aligned real/fake waveform segments per sample for adversarial losses."""
    if crop_samples <= 0:
        return wav, wav_hat
    T = min(wav.shape[1], wav_hat.shape[1])
    wav = wav[:, :T]
    wav_hat = wav_hat[:, :T]
    if T <= crop_samples:
        return wav, wav_hat
    starts = torch.randint(0, T - crop_samples + 1, (wav.shape[0],), device=wav.device)
    offsets = torch.arange(crop_samples, device=wav.device).unsqueeze(0)
    idx = starts.unsqueeze(1) + offsets
    return torch.gather(wav, 1, idx), torch.gather(wav_hat, 1, idx)


def _maybe_clip_grad_norm_(parameters, max_norm: float | None) -> torch.Tensor:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    if max_norm is not None and max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    norms = [p.grad.detach().norm(2) for p in params]
    return torch.linalg.vector_norm(torch.stack(norms), 2)


if __name__ == "__main__":
    main()
