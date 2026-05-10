"""TTL training (Stage 2): text-to-latent flow matching, paper-faithful.

Paper recipe (arXiv 2503.23108):
  L_TTL = E[ || m ⊙ ( v_θ(z_t, z_ref, c, t) - (z_1 - (1-σ_min)·z_0) ) ||_1 ]

Paper-faithful settings (defaults):
  AdamW, lr=5e-4, halve every 300k, 700k iter, batch=64, K_e=4
  σ_min = 1e-8, p_uncond = 0.05 (joint dropout)
  Reference encoder: random crop 0.2-9 s (≤ ½ utt duration)
  Loss mask m: 1 OUTSIDE the reference crop, 0 INSIDE — prevents leakage
  Channel-wise z-score latent normalization (no extra scale)

Paper architecture (defaults):
  VectorField: dim=256, style_dim=128, text_dim=128, intermediate=1024, ksz=5
  StyleEncoder: hdim=128, value_dim=128 (StyleEncoderTTLPaper)
  TextEncoder: dim=128, shared 50x128 reference key

KSS-adapted (RTX 3090 × 1):
  batch=8, K_e=4 → effective 32 vs paper 64*4=256

Run:
  # Prereq: AE trained + latents cached:
  #   python -m training.scripts.cache_latents --ckpt ... --out_dir ./cache
  python -m training.scripts.train_ttl --cache_dir ./cache --steps 700000
"""
from __future__ import annotations
import os, sys, json, argparse, time, random
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
from training.models.style_encoder import StyleEncoderTTL, StyleEncoderTTLPaper
from training.losses.flow_matching import (
    sample_flow_matching_inputs, cfm_loss,
    UncondMasker, UncondMaskerConfig, batch_expand,
    SIGMA_MIN,
)

from torch_text_encoder import TextEncoder, TextEncoderPaper, load_text_encoder_weights   # type: ignore
from torch_vector_estimator import VectorField, load_ve_weights         # type: ignore


@dataclass
class TTLConfig:
    # data
    cache_dir: str = ""
    unicode_indexer: str = "assets/onnx/unicode_indexer.json"
    lang: str = "ko"
    # mode: paper-faithful from-scratch vs shipped fine-tune
    paper_faithful: bool = True   # if True: paper dims (VF=256, style=128); else shipped (VF=512, style=256)
    fine_tune: bool = False       # if True: load shipped TE+VF, freeze, train only style_encoder
    te_onnx: str = "assets/onnx/text_encoder.onnx"
    ve_onnx: str = "assets/onnx/vector_estimator.onnx"
    # optim (paper B.2)
    lr: float = 5e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    grad_clip: float | None = None   # paper does not specify grad clipping; default off
    # schedule
    steps: int = 700_000          # OPTIMIZER UPDATES (matches paper iter count)
    lr_halve_every: int = 300_000
    batch_size: int = 8           # microbatch per forward
    k_expand: int = 4             # paper Sec 3.2.3: K_e=4
    grad_accum: int = 1           # microbatches per optimizer update; effective batch = batch_size * grad_accum
    attn_type: str = "rope"       # paper A.2.2: text self-attention uses RoPE
    num_workers: int = 2
    # CFG dropout (paper B.2: single joint p_uncond=0.05)
    prob_uncond: float = 0.05
    # logging
    log_every: int = 50
    ckpt_every: int = 50_000
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
    ap.add_argument("--k_expand", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--shipped_dim", action="store_true",
                    help="use shipped 512-dim arch instead of paper 256-dim (for fine-tune)")
    ap.add_argument("--fine_tune", action="store_true",
                    help="load shipped TE+VF and freeze, train only style_encoder")
    ap.add_argument("--ckpt_every", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=None,
                    help="microbatches per optimizer update (effective batch = batch_size * grad_accum)")
    ap.add_argument("--attn_type", type=str, default="rope", choices=["rope", "relpos"],
                    help="text encoder self-attn position encoding (paper: rope)")
    ap.add_argument("--grad_clip", type=float, default=None,
                    help="optional grad-norm clip; default off (paper unspecified)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = TTLConfig(cache_dir=args.cache_dir)
    if args.steps is not None:      cfg.steps = args.steps
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.k_expand is not None:   cfg.k_expand = args.k_expand
    if args.out_dir is not None:    cfg.out_dir = args.out_dir
    if args.resume is not None:     cfg.resume = args.resume
    if args.shipped_dim:            cfg.paper_faithful = False
    if args.fine_tune:              cfg.fine_tune = True; cfg.paper_faithful = False
    if args.ckpt_every is not None: cfg.ckpt_every = args.ckpt_every
    if args.log_every is not None:  cfg.log_every = args.log_every
    if args.num_workers is not None: cfg.num_workers = args.num_workers
    if args.grad_accum is not None: cfg.grad_accum = args.grad_accum
    if args.attn_type is not None:  cfg.attn_type = args.attn_type
    if args.grad_clip is not None:  cfg.grad_clip = args.grad_clip
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
    print(f"[info] paper_faithful={cfg.paper_faithful}  fine_tune={cfg.fine_tune}  σ_min={SIGMA_MIN}")

    # --- data ---
    ds, dl = make_loader(cfg)
    print(f"[info] dataset: {len(ds)} utterances")
    data_iter = iter(_inf_loader(dl))

    # --- models ---
    if cfg.paper_faithful:
        # paper A.2.1 / A.2.2 / A.2.3: text_dim=128, style_dim=128, VF dim=256
        text_encoder = TextEncoderPaper(style_dim=128, attn_type=cfg.attn_type).to(device)
        style_encoder = StyleEncoderTTLPaper().to(device)
        vector_field = VectorField(
            dim=256, latent_dim=144, n_outer=4, time_dim=64,
            inter=1024, ksz=5, text_dim=128, style_dim=128,
            learn_style_prototype=False,
        ).to(device)
        text_dim = 128
        style_value_dim = 128
    else:
        # shipped 512-dim arch (for fine-tune of shipped weights)
        text_encoder = TextEncoder().to(device)   # default style_dim=256
        style_encoder = StyleEncoderTTL().to(device)
        vector_field = VectorField().to(device)   # default dim=512, style_dim=256
        text_dim = 256
        style_value_dim = 256
    uncond_masker = UncondMasker(UncondMaskerConfig(
        prob_uncond=cfg.prob_uncond,
        text_dim=text_dim,
        n_style=50,
        style_value_dim=style_value_dim,
    )).to(device)

    if cfg.fine_tune:
        load_text_encoder_weights(text_encoder, cfg.te_onnx)
        load_ve_weights(vector_field, cfg.ve_onnx)
        for p in text_encoder.parameters(): p.requires_grad_(False)
        for p in vector_field.parameters(): p.requires_grad_(False)
        text_encoder.eval(); vector_field.eval()
        trainable = list(style_encoder.parameters()) + list(uncond_masker.parameters())
        print(f"[info] fine-tune: shipped TE+VF frozen, train only style_encoder + uncond_masker")
    else:
        trainable = (list(text_encoder.parameters()) + list(style_encoder.parameters())
                     + list(vector_field.parameters()) + list(uncond_masker.parameters()))
        print(f"[info] from-scratch: training all modules")

    n_te = sum(p.numel() for p in text_encoder.parameters())
    n_se = sum(p.numel() for p in style_encoder.parameters())
    n_vf = sum(p.numel() for p in vector_field.parameters())
    n_um = sum(p.numel() for p in uncond_masker.parameters())
    n_train = sum(p.numel() for p in trainable)
    print(f"[info] params: TE={n_te/1e6:.2f}M  SE={n_se/1e6:.2f}M  VF={n_vf/1e6:.2f}M  UM={n_um/1e6:.2f}M  | trainable={n_train/1e6:.2f}M")

    opt = torch.optim.AdamW(trainable, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                            weight_decay=cfg.weight_decay)
    def lr_at(step):
        return cfg.lr * (0.5 ** (step // cfg.lr_halve_every))

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

    style_encoder.train(); uncond_masker.train()
    if not cfg.fine_tune:
        text_encoder.train(); vector_field.train()
    t_start = time.time()

    accum = max(1, cfg.grad_accum)
    print(f"[info] grad_accum={accum}  "
          f"(effective batch = batch_size {cfg.batch_size} × accum {accum} = {cfg.batch_size * accum} unique samples per optimizer update)")

    for step in range(step0 + 1, cfg.steps + 1):
        # One outer step = `accum` microbatches followed by ONE optimizer update.
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0   # mean loss across microbatches (for logging)
        for micro in range(accum):
            batch = next(data_iter)
            z_1            = batch["z_ttl"].to(device, non_blocking=True)
            latent_mask    = batch["latent_mask"].to(device, non_blocking=True)
            ref_loss_mask  = batch["ref_loss_mask"].to(device, non_blocking=True)
            z_ae_ref       = batch["z_ae_ref"].to(device, non_blocking=True)
            ref_frame_mask = batch["ref_frame_mask"].to(device, non_blocking=True)
            text_ids       = batch["text_ids"].to(device, non_blocking=True)
            text_mask      = batch["text_mask"].to(device, non_blocking=True)

            # paper m: loss only on non-reference, non-pad TTL frames
            loss_mask = latent_mask * ref_loss_mask

            # Reference encoder receives the CROPPED latent (paper Sec 3.2.4)
            style_ttl = style_encoder(z_ae_ref, ref_frame_mask)
            text_emb  = text_encoder(text_ids, style_ttl, text_mask)

            # CFG dropout (single joint p_uncond=0.05, paper B.2)
            text_emb_m, style_ttl_m = uncond_masker(text_emb, style_ttl)

            # K_e batch expand (paper Sec 3.2.3)
            K = cfg.k_expand
            z1_e, lm_e, lossm_e, te_e, st_e, tm_e = batch_expand(
                z_1, latent_mask, loss_mask, text_emb_m, style_ttl_m, text_mask, K=K
            )
            z_t, target, _z0, t = sample_flow_matching_inputs(z1_e, lm_e)

            style_prototype = text_encoder.reference_key if cfg.paper_faithful else None
            v_pred = vector_field.velocity(
                z_t, te_e, st_e, lm_e, tm_e, t,
                style_prototype=style_prototype,
            )
            micro_loss = cfm_loss(v_pred, target, lossm_e)

            # Scale for grad accumulation; sum-of-mean over microbatches gives correct mean grad.
            (micro_loss / accum).backward()
            loss_acc += micro_loss.item() / accum

        clip_at = cfg.grad_clip if cfg.grad_clip is not None else float("inf")
        gn = torch.nn.utils.clip_grad_norm_(trainable, clip_at)
        lr_now = lr_at(step)
        for pg in opt.param_groups: pg["lr"] = lr_now
        opt.step()

        if step % cfg.log_every == 0:
            elapsed = time.time() - t_start
            sps = (step - step0) / max(elapsed, 1e-9)
            print(f"[step {step}/{cfg.steps}]  loss={loss_acc:.4f}  "
                  f"gn={gn:.2f}  lr={lr_now:.2e}  |  {sps:.2f} upd/s",
                  flush=True)
            if tb is not None:
                tb.add_scalar("loss/cfm", loss_acc, step)
                tb.add_scalar("grad_norm", gn.item(), step)
                tb.add_scalar("lr", lr_now, step)
                tb.add_scalar("sys/upd_per_sec", sps, step)

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
