"""TTL training (Stage 2): text-to-latent flow matching.

Paper recipe (arXiv 2503.23108):
  L_TTL = E[ || m ⊙ ( v_θ(z_t, z_ref, c, t) - (z_1 - (1-σ)·z_0) ) ||_1 ]
  AdamW, lr=5e-4, halve every 300k, batch=64 × K_e=6, 700k steps (paper).

KSS-adapted (RTX 3090 × 1):
  batch=8, K_e=6 → effective batch 48, ~100-200k steps.

Run:
  # Prereq: AE trained + latents cached:
  #   python -m training.scripts.cache_latents --ckpt ... --out_dir ./cache
  #
  # Then:
  python -m training.scripts.train_ttl --cache_dir ./cache --smoke
  python -m training.scripts.train_ttl --cache_dir ./cache --steps 150000
"""
from __future__ import annotations
import os, sys, json, argparse, time
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from training.data.ttl_dataset import TTLDataset, collate_ttl, TTL_NORMALIZER_SCALE
from training.models.style_encoder import StyleEncoderTTL
from training.losses.flow_matching import (
    sample_flow_matching_inputs, cfm_loss,
    UncondMasker, UncondMaskerConfig, batch_expand,
)

# Import verified inference modules from analysis/
from torch_text_encoder import TextEncoder, load_text_encoder_weights   # type: ignore
from torch_vector_estimator import VectorField, load_ve_weights         # type: ignore


@dataclass
class TTLConfig:
    # data
    cache_dir: str = ""
    unicode_indexer: str = "assets/onnx/unicode_indexer.json"
    lang: str = "ko"
    # fine-tune: load shipped text_encoder + vector_estimator as init and freeze.
    # This collapses training to just StyleEncoderTTL (+ uncond_masker tokens) —
    # 1.5M params instead of 44M — and makes convergence tractable on small data.
    fine_tune: bool = True
    te_onnx: str = "assets/onnx/text_encoder.onnx"
    ve_onnx: str = "assets/onnx/vector_estimator.onnx"
    # optim
    lr: float = 5e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    # schedule
    steps: int = 700_000
    lr_halve_every: int = 300_000
    batch_size: int = 8
    k_expand: int = 6
    num_workers: int = 2
    # CFG dropout
    prob_both_uncond: float = 0.04
    prob_text_uncond: float = 0.01
    uncond_std: float = 0.1
    # logging
    log_every: int = 50
    ckpt_every: int = 20_000
    out_dir: str = "training/runs/ttl"
    resume: str | None = None


def make_loader(cfg: TTLConfig):
    ds = TTLDataset(cfg.cache_dir, cfg.unicode_indexer, lang=cfg.lang)
    mean = ds.mean; std = ds.std
    coll = partial(collate_ttl, mean=mean, std=std, kc=6, scale=TTL_NORMALIZER_SCALE)
    dl = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True, pin_memory=True,
        collate_fn=coll, persistent_workers=cfg.num_workers > 0,
    )
    return ds, dl


def _inf_loader(dl):
    while True:
        for b in dl:
            yield b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--from_scratch", action="store_true",
                    help="disable fine-tune mode; train text_encoder + vector_estimator from random init")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = TTLConfig(cache_dir=args.cache_dir)
    if args.steps is not None:      cfg.steps = args.steps
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.out_dir is not None:    cfg.out_dir = args.out_dir
    if args.resume is not None:     cfg.resume = args.resume
    if args.from_scratch:           cfg.fine_tune = False
    if args.smoke:
        cfg.steps = 300
        cfg.batch_size = 2
        cfg.k_expand = 2
        cfg.log_every = 10
        cfg.ckpt_every = 10_000
        cfg.out_dir = "training/runs/ttl_smoke"

    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device(args.device)
    print(f"[info] device: {device}")

    # --- data ---
    ds, dl = make_loader(cfg)
    print(f"[info] dataset: {len(ds)} utterances")
    data_iter = iter(_inf_loader(dl))

    # z-score stats for normalizing AE latent before the style encoder.
    # Raw z_ae from the (undertrained) AE has arbitrary scale (std ~100s~1000s);
    # feeding it unnormalized makes style_encoder outputs explode (std 1e4+) and
    # kills downstream gradients. We standardize here so style_encoder sees ~N(0,1).
    _ae_mean = ds.mean.to(device).view(1, -1, 1)
    _ae_std  = ds.std.to(device).view(1, -1, 1)

    # --- models ---
    text_encoder = TextEncoder().to(device)
    style_encoder = StyleEncoderTTL().to(device)
    vector_field = VectorField().to(device)
    uncond_masker = UncondMasker(UncondMaskerConfig(
        prob_both_uncond=cfg.prob_both_uncond,
        prob_text_uncond=cfg.prob_text_uncond,
        std=cfg.uncond_std,
    )).to(device)

    # Fine-tune mode: load shipped weights into text_encoder + vector_field, freeze them.
    # StyleEncoderTTL starts random and is the only "real" trainable module; adapts a
    # new speaker/dataset into the pre-trained conditioning pipeline.
    if cfg.fine_tune:
        load_text_encoder_weights(text_encoder, cfg.te_onnx)
        load_ve_weights(vector_field, cfg.ve_onnx)
        for p in text_encoder.parameters(): p.requires_grad_(False)
        for p in vector_field.parameters(): p.requires_grad_(False)
        text_encoder.eval()
        vector_field.eval()
        trainable = list(style_encoder.parameters()) + list(uncond_masker.parameters())
        print(f"[info] fine-tune mode: shipped weights loaded, TE+VF frozen.")
    else:
        trainable = (list(text_encoder.parameters()) + list(style_encoder.parameters())
                     + list(vector_field.parameters()) + list(uncond_masker.parameters()))
        print(f"[info] from-scratch mode: training all modules.")

    all_params = trainable
    n_params = sum(p.numel() for p in all_params)
    print(f"[info] trainable params: {n_params/1e6:.2f}M")

    # --- optim ---
    opt = torch.optim.AdamW(all_params,
                            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                            weight_decay=cfg.weight_decay)
    # LR schedule: halve every N steps
    def lr_at(step):
        return cfg.lr * (0.5 ** (step // cfg.lr_halve_every))

    # --- resume ---
    step0 = 0
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=device, weights_only=False)
        text_encoder.load_state_dict(ck["text_encoder"])
        style_encoder.load_state_dict(ck["style_encoder"])
        vector_field.load_state_dict(ck["vector_field"])
        uncond_masker.load_state_dict(ck["uncond_masker"])
        opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        print(f"[info] resumed from step {step0}")

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(out_dir / "tb"))
    except Exception:
        tb = None

    # --- train ---
    # In fine-tune mode, keep frozen modules in eval (no grad, no train-mode drift).
    style_encoder.train(); uncond_masker.train()
    if not cfg.fine_tune:
        text_encoder.train(); vector_field.train()
    t_start = time.time()
    for step in range(step0 + 1, cfg.steps + 1):
        batch = next(data_iter)
        z_1       = batch["z_ttl"].to(device, non_blocking=True)       # [B, 144, L]
        lat_mask  = batch["latent_mask"].to(device, non_blocking=True) # [B, 1, L]
        z_ae      = batch["z_ae"].to(device, non_blocking=True)        # [B, 24, T_ae]
        text_ids  = batch["text_ids"].to(device, non_blocking=True)    # [B, T_text]
        text_mask = batch["text_mask"].to(device, non_blocking=True)   # [B, 1, T_text]
        B = z_1.shape[0]

        # --- condition encoders (run once per original batch, will be expanded) ---
        z_ae_norm = (z_ae - _ae_mean) / _ae_std                      # z-score to ~N(0,1)
        style_ttl = style_encoder(z_ae_norm)                         # [B, 50, 256]
        text_emb  = text_encoder(text_ids, style_ttl, text_mask)     # [B, 256, T_text]

        # CFG dropout
        text_emb_m, style_ttl_m = uncond_masker(text_emb, style_ttl)

        # --- batch expand for K different (z_0, t) per sample ---
        K = cfg.k_expand
        z1_e, lm_e, te_e, st_e, tm_e = batch_expand(
            z_1, lat_mask, text_emb_m, style_ttl_m, text_mask, K=K
        )

        # Sample z_t, target velocity
        z_t, target, _z0, t = sample_flow_matching_inputs(z1_e, lm_e)

        # Velocity prediction
        v_pred = vector_field.velocity(z_t, te_e, st_e, lm_e, tm_e, t)
        loss = cfm_loss(v_pred, target, lm_e)

        # --- backward ---
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
        # update LR
        lr_now = lr_at(step)
        for pg in opt.param_groups: pg["lr"] = lr_now
        opt.step()

        if step % cfg.log_every == 0:
            elapsed = time.time() - t_start
            sps = (step - step0) / max(elapsed, 1e-9)
            print(f"[step {step}/{cfg.steps}]  loss={loss.item():.4f}  "
                  f"gn={gn:.2f}  lr={lr_now:.2e}  |  {sps:.2f} step/s",
                  flush=True)
            if tb is not None:
                tb.add_scalar("loss/cfm", loss.item(), step)
                tb.add_scalar("grad_norm", gn.item(), step)
                tb.add_scalar("lr", lr_now, step)
                tb.add_scalar("sys/step_per_sec", sps, step)

        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            ck_path = out_dir / f"ckpt_step{step:08d}.pt"
            torch.save({
                "step": step,
                "text_encoder":  text_encoder.state_dict(),
                "style_encoder": style_encoder.state_dict(),
                "vector_field":  vector_field.state_dict(),
                "uncond_masker": uncond_masker.state_dict(),
                "opt": opt.state_dict(),
                "cfg": asdict(cfg),
            }, ck_path)
            print(f"[ckpt] {ck_path}")

    print(f"[done] total {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
