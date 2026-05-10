"""DP training (Stage 3): duration predictor.

Paper recipe (arXiv 2503.23108):
  L = L1(pred_duration_sec, gt_duration_sec)   (utterance-level scalar)
  AdamW, lr=5e-4, batch=128, 3,000 iter
  Reference encoder receives a 5%-95% random crop of input speech (paper Sec 4.2)

Prereqs: AE trained + latents cached (same cache as TTL).

Run:
  python -m training.scripts.train_dp --cache_dir ./cache --steps 3000
"""
from __future__ import annotations
import os, sys, json, argparse, time
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "py"))

from training.data.ttl_dataset import TTLDataset, collate_dp, TTL_NORMALIZER_SCALE
from training.models.style_encoder import StyleEncoderDP, StyleEncoderDPPaper, StyleEncoderDPTextPaper

from torch_duration_predictor import (
    DurationPredictor, DurationPredictorPaper, load_dp_weights,
)  # type: ignore


@dataclass
class DPConfig:
    cache_dir: str = ""
    unicode_indexer: str = "assets/onnx/unicode_indexer.json"
    lang: str = "ko"
    # mode: paper-faithful from-scratch vs shipped fine-tune
    paper_faithful: bool = True   # use StyleEncoderDPPaper (out_scale=1.0)
    paper_text_estimator: bool = False  # experimental: interpret paper's inconsistent DP dim text literally-ish
    fine_tune: bool = False        # if True: load shipped DP, freeze, train only style enc
    dp_onnx: str = "assets/onnx/duration_predictor.onnx"
    attn_type: str = "rope"        # paper A.3.2: DP sentence encoder self-attn uses RoPE
    # optim (paper Sec 4.2)
    lr: float = 5e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    grad_clip: float | None = None   # paper doesn't specify; default off
    # schedule
    steps: int = 3000              # paper: 3,000 iter
    batch_size: int = 128          # paper: 128
    num_workers: int = 2
    log_every: int = 20
    ckpt_every: int = 500
    out_dir: str = "training/runs/dp"
    resume: str | None = None


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
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--fine_tune", action="store_true",
                    help="load shipped DP and freeze; train only style enc")
    ap.add_argument("--shipped_dim", action="store_true",
                    help="use shipped StyleEncoderDP (out_scale=0.0625) instead of paper")
    ap.add_argument("--paper_text_estimator", action="store_true",
                    help="experimental: use inferred 64-ref / 128-input DP estimator despite ONNX mismatch")
    ap.add_argument("--attn_type", type=str, default=None, choices=["rope", "relpos"],
                    help="DP sentence encoder self-attn position encoding (paper: rope; default rope for paper-faithful, relpos for fine-tune)")
    ap.add_argument("--ckpt_every", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=None)
    ap.add_argument("--grad_clip", type=float, default=None,
                    help="optional grad-norm clip; default off (paper unspecified)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = DPConfig(cache_dir=args.cache_dir)
    if args.steps is not None:      cfg.steps = args.steps
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.out_dir is not None:    cfg.out_dir = args.out_dir
    if args.resume is not None:     cfg.resume = args.resume
    if args.num_workers is not None: cfg.num_workers = args.num_workers
    if args.fine_tune:              cfg.fine_tune = True; cfg.paper_faithful = False
    if args.shipped_dim:            cfg.paper_faithful = False
    if args.paper_text_estimator:   cfg.paper_text_estimator = True
    if args.attn_type is not None:  cfg.attn_type = args.attn_type
    # Fine-tune path loads shipped ONNX weights which were trained with relpos;
    # force relpos there so load_dp_weights matches the parameter layout.
    if cfg.fine_tune:               cfg.attn_type = "relpos"
    if args.ckpt_every is not None: cfg.ckpt_every = args.ckpt_every
    if args.log_every is not None:  cfg.log_every = args.log_every
    if args.grad_clip is not None:  cfg.grad_clip = args.grad_clip
    if args.smoke:
        cfg.steps = 200
        cfg.batch_size = 4
        cfg.log_every = 10
        cfg.ckpt_every = 1000
        cfg.out_dir = "training/runs/dp_smoke"

    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    device = torch.device(args.device)
    print(f"[info] device: {device}")
    print(f"[info] paper_faithful={cfg.paper_faithful}  fine_tune={cfg.fine_tune}")

    ds = TTLDataset(cfg.cache_dir, cfg.unicode_indexer, lang=cfg.lang)
    coll = partial(collate_dp, mean=ds.mean, std=ds.std, scale=TTL_NORMALIZER_SCALE)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, drop_last=True, pin_memory=True,
                    collate_fn=coll, persistent_workers=cfg.num_workers > 0)
    data_iter = iter(_inf_loader(dl))
    print(f"[info] dataset: {len(ds)} utterances")

    # models
    print(f"[info] attn_type={cfg.attn_type}")
    if cfg.paper_faithful:
        if cfg.paper_text_estimator:
            # Experimental reading of paper A.3.1/A.3.3. The PDF's dimensions are
            # inconsistent, so this is not the default.
            style_enc = StyleEncoderDPTextPaper().to(device)      # outputs [B, 64]
            dp = DurationPredictorPaper(attn_type=cfg.attn_type).to(device)
        else:
            # Default: use the deployed ONNX estimator shape (64 text + 8*16 ref).
            style_enc = StyleEncoderDPPaper().to(device)          # outputs [B, 8, 16]
            dp = DurationPredictor(attn_type=cfg.attn_type).to(device)
    else:
        # Shipped: ref [8,16]=128 + text 64 → estimator(192→128→1)
        style_enc = StyleEncoderDP().to(device)                    # outputs [B, 8, 16]
        dp = DurationPredictor(attn_type=cfg.attn_type).to(device)

    if cfg.fine_tune:
        load_dp_weights(dp, cfg.dp_onnx)
        for p in dp.parameters(): p.requires_grad_(False)
        dp.eval()
        params = list(style_enc.parameters())
        print(f"[info] fine-tune mode: shipped DP frozen, training style_enc only")
    else:
        params = list(dp.parameters()) + list(style_enc.parameters())
        print(f"[info] from-scratch mode: training DP + style_enc")
    n = sum(p.numel() for p in params)
    print(f"[info] trainable params: {n/1e6:.2f}M")

    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                            weight_decay=cfg.weight_decay)

    step0 = 0
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=device, weights_only=False)
        dp.load_state_dict(ck["dp"]); style_enc.load_state_dict(ck["style_enc"])
        opt.load_state_dict(ck["opt"]); step0 = ck["step"]
        print(f"[info] resumed from step {step0}")

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(out_dir / "tb"))
    except Exception:
        tb = None

    style_enc.train()
    if not cfg.fine_tune:
        dp.train()
    t_start = time.time()
    for step in range(step0 + 1, cfg.steps + 1):
        batch = next(data_iter)
        text_ids          = batch["text_ids"].to(device, non_blocking=True)
        text_mask         = batch["text_mask"].to(device, non_blocking=True)
        z_ae_ref_dp       = batch["z_ae_ref_dp"].to(device, non_blocking=True)
        ref_dp_frame_mask = batch["ref_dp_frame_mask"].to(device, non_blocking=True)
        gt_dur            = batch["gt_duration_sec"].to(device, non_blocking=True)

        style_dp = style_enc(z_ae_ref_dp, ref_dp_frame_mask)         # [B, 8, 16]
        pred_dur = dp(text_ids, style_dp, text_mask)                  # [B]

        loss = torch.abs(pred_dur - gt_dur).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # measure grad norm regardless; clip only if grad_clip is set (paper: unspecified)
        clip_at = cfg.grad_clip if cfg.grad_clip is not None else float("inf")
        gn = torch.nn.utils.clip_grad_norm_(params, clip_at)
        opt.step()

        if step % cfg.log_every == 0:
            elapsed = time.time() - t_start
            sps = (step - step0) / max(elapsed, 1e-9)
            mae = loss.item()
            rel = mae / gt_dur.mean().item()
            print(f"[step {step}/{cfg.steps}]  L1={mae:.4f}s  "
                  f"rel={100*rel:.1f}%  gn={gn:.2f}  |  {sps:.2f} step/s",
                  flush=True)
            if tb is not None:
                tb.add_scalar("loss/L1", mae, step)
                tb.add_scalar("loss/relative", rel, step)
                tb.add_scalar("grad_norm", gn.item(), step)

        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            ck_path = out_dir / f"ckpt_step{step:08d}.pt"
            torch.save({
                "step": step,
                "dp": dp.state_dict(),
                "style_enc": style_enc.state_dict(),
                "opt": opt.state_dict(),
                "cfg": asdict(cfg),
            }, ck_path)
            print(f"[ckpt] {ck_path}")

    print(f"[done] total {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
