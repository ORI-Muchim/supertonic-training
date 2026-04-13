"""DP training (Stage 3): duration predictor.

Paper recipe (arXiv 2503.23108):
  L = L1(pred_duration_sec, gt_duration_sec)  (utterance-level scalar)
  AdamW, lr=5e-4, batch=128, ~3k steps. Trivial.

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

from training.data.ttl_dataset import TTLDataset, collate_ttl, TTL_NORMALIZER_SCALE
from training.models.style_encoder import StyleEncoderDP

from torch_duration_predictor import DurationPredictor  # type: ignore


@dataclass
class DPConfig:
    cache_dir: str = ""
    unicode_indexer: str = "assets/onnx/unicode_indexer.json"
    lang: str = "ko"
    lr: float = 5e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    steps: int = 3000
    batch_size: int = 32
    num_workers: int = 2
    log_every: int = 20
    ckpt_every: int = 1000
    out_dir: str = "training/runs/dp"
    resume: str | None = None


def _get_durations(batch, sample_rate: int = 44100, hop: int = 512):
    """Recover ground-truth duration (seconds) from the TTL dataset item.
    frame_mask tells us actual T_ae frames → duration in samples ≈ T_ae * hop.
    """
    # frame_mask: [B, 1, T_ae_max]
    T_ae = batch["frame_mask"].sum(dim=(-2, -1))  # [B]  number of valid AE frames
    # Note: AE uses center=True STFT so output is exactly T_ae * hop samples for the decoder.
    duration = T_ae * hop / sample_rate
    return duration


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
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = DPConfig(cache_dir=args.cache_dir)
    if args.steps is not None:      cfg.steps = args.steps
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.out_dir is not None:    cfg.out_dir = args.out_dir
    if args.resume is not None:     cfg.resume = args.resume
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

    ds = TTLDataset(cfg.cache_dir, cfg.unicode_indexer, lang=cfg.lang)
    coll = partial(collate_ttl, mean=ds.mean, std=ds.std)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, drop_last=True, pin_memory=True,
                    collate_fn=coll, persistent_workers=cfg.num_workers > 0)
    data_iter = iter(_inf_loader(dl))
    print(f"[info] dataset: {len(ds)} utterances")

    # models
    dp = DurationPredictor().to(device)
    style_enc = StyleEncoderDP().to(device)

    params = list(dp.parameters()) + list(style_enc.parameters())
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

    dp.train(); style_enc.train()
    t_start = time.time()
    for step in range(step0 + 1, cfg.steps + 1):
        batch = next(data_iter)
        z_ae      = batch["z_ae"].to(device, non_blocking=True)
        text_ids  = batch["text_ids"].to(device, non_blocking=True)
        text_mask = batch["text_mask"].to(device, non_blocking=True)
        gt_dur    = _get_durations(batch).to(device, non_blocking=True)

        style_dp = style_enc(z_ae)                        # [B, 8, 16]
        pred_dur = dp(text_ids, style_dp, text_mask)       # [B]

        loss = torch.abs(pred_dur - gt_dur).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()

        if step % cfg.log_every == 0:
            elapsed = time.time() - t_start
            sps = (step - step0) / max(elapsed, 1e-9)
            mae = loss.item()
            # Relative error = MAE / mean_gt_duration
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
