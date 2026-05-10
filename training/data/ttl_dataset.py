"""Dataset for TTL (Stage 2) training — loads cached AE latents + text_ids.

Paper-faithful (arXiv 2503.23108, Sec 3.2.4 / B.2):
  - Reference encoder receives a RANDOM CROP of the AE latent (0.2-9 s, ≤ ½ utt duration).
  - Reference loss mask m: 1 OUTSIDE the crop, 0 INSIDE — applied to L_CFM so the
    flow-matching loss is computed only on non-reference frames (prevents leakage).
  - Latent normalization is single z-score (channel-wise mean/std), no extra scale.
"""
from __future__ import annotations
import os, sys, json, math, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "py"))
from helper import UnicodeProcessor  # type: ignore


# Paper does not specify any extra multiplicative scale beyond channel-wise z-score.
# Set to 1.0 for paper-faithful from-scratch training.
TTL_NORMALIZER_SCALE = 1.0

# Paper Sec 3.2.4 reference crop duration bounds (in AE frames at 44.1 kHz / hop 512):
SR = 44100
HOP = 512
REF_CROP_MIN_SEC = 0.2
REF_CROP_MAX_SEC = 9.0
REF_CROP_MIN_FRAMES = max(1, int(round(REF_CROP_MIN_SEC * SR / HOP)))   # 17
REF_CROP_MAX_FRAMES = int(round(REF_CROP_MAX_SEC * SR / HOP))           # 775


def prepare_ttl_latent(
    z_ae: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    scale: float = TTL_NORMALIZER_SCALE,
    kc: int = 6,
) -> torch.Tensor:
    """Convert raw AE latent [C, T] → normalized chunk-compressed TTL latent [C*kc, T/kc]."""
    mean = mean.to(z_ae.device).view(-1, 1)
    std  = std.to(z_ae.device).view(-1, 1)
    z_norm = (z_ae - mean) / std * scale
    C, T = z_norm.shape
    if T % kc != 0:
        pad = kc - (T % kc)
        z_norm = torch.cat([z_norm, z_norm[:, -1:].expand(-1, pad)], dim=-1)
        T = z_norm.shape[-1]
    z_norm = z_norm.reshape(C, T // kc, kc).permute(0, 2, 1).contiguous()
    z_ttl = z_norm.reshape(C * kc, T // kc)
    return z_ttl


def invert_ttl_latent(
    z_ttl: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    scale: float = TTL_NORMALIZER_SCALE,
    kc: int = 6,
) -> torch.Tensor:
    """Inverse: [C*kc, T_ttl] → raw AE latent [C, T_ae]."""
    Ckc, T_ttl = z_ttl.shape
    assert Ckc % kc == 0
    C = Ckc // kc
    z = z_ttl.reshape(C, kc, T_ttl).permute(0, 2, 1).reshape(C, T_ttl * kc)
    mean = mean.to(z.device).view(-1, 1)
    std  = std.to(z.device).view(-1, 1)
    return (z / scale) * std + mean


def sample_reference_crop(T_ae: int, kc: int = 6, rng: random.Random | None = None) -> tuple[int, int]:
    """Sample a reference crop window in AE-frame indices.

    Paper Sec 3.2.4:
      duration ∈ [0.2 s, min(9 s, T_ae/2)]
    Returns (start, end) inclusive-exclusive AE-frame indices.

    For loss-mask alignment (TTL frame = kc AE frames), we round the crop boundaries
    to multiples of kc so the ref crop maps cleanly to TTL-frame indices.
    """
    rng = rng or random
    half_frames = max(1, T_ae // 2)
    raw_max = min(REF_CROP_MAX_FRAMES, half_frames)
    raw_min = min(REF_CROP_MIN_FRAMES, raw_max)
    # Sample in kc-aligned units. Use ceil for the lower bound so ordinary
    # utterances never slip below the paper's 0.2 s minimum after alignment.
    if raw_max < kc:
        # Utterance shorter than 2 × min crop: cap crop at half
        crop_len = raw_max
    else:
        min_units = max(1, math.ceil(raw_min / kc))
        max_units = max(1, raw_max // kc)
        if min_units > max_units:
            min_units = max_units
        crop_len = rng.randint(min_units, max_units) * kc
    start_max = T_ae - crop_len
    if start_max <= 0:
        start = 0
    else:
        start = rng.randint(0, start_max)
        # round start to multiple of kc as well
        start = (start // kc) * kc
    end = start + crop_len
    return start, end


class TTLDataset(Dataset):
    """Returns one utterance per __getitem__ (variable length — collate with padding)."""
    def __init__(
        self,
        cache_dir: str | Path,
        unicode_indexer_path: str | Path,
        lang: str = "ko",
    ):
        cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        with open(cache_dir / "manifest.json", "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.stats = torch.load(cache_dir / "stats.pt", weights_only=False)
        self.mean = self.stats["mean"].float()
        self.std  = self.stats["std"].float()
        self.text_processor = UnicodeProcessor(str(unicode_indexer_path))
        self.lang = lang

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, i):
        m = self.manifest[i]
        z_ae = torch.load(self.cache_dir / m["latent_path"], weights_only=False).float()  # [24, T]
        text_ids_np, _ = self.text_processor([m["text_norm"]], [self.lang])
        text_ids = torch.from_numpy(text_ids_np[0])
        return {
            "z_ae": z_ae,
            "text_ids": text_ids,
            "text_len": int(text_ids.shape[0]),
            "lang": self.lang,
            "text_raw": m["text_raw"],
            "T_ae": int(z_ae.shape[-1]),
        }


def collate_ttl(batch: list[dict], mean: torch.Tensor, std: torch.Tensor,
                kc: int = 6, scale: float = TTL_NORMALIZER_SCALE,
                rng: random.Random | None = None):
    """Paper-faithful collate with reference cropping + reference loss mask.

    Returns:
        z_ttl        : [B, 144, L_max]   chunk-compressed full latent (target z_1)
        latent_mask  : [B, 1, L_max]     padding mask in TTL-frame space (1=valid, 0=pad)
        ref_loss_mask: [B, 1, L_max]     paper m: 1 OUTSIDE ref crop, 0 INSIDE crop AND pad
        z_ae_ref     : [B, 24, T_ref_max]  CROPPED AE latent for reference encoder
        ref_frame_mask: [B, 1, T_ref_max]  AE-frame mask within the crop (for attn masking)
        text_ids     : [B, T_max]
        text_mask    : [B, 1, T_max]
    """
    rng = rng or random
    B = len(batch)
    T_text_max = max(b["text_len"] for b in batch)
    L_ttl_max = max((b["T_ae"] + kc - 1) // kc for b in batch)

    # First pass: sample reference crops per sample
    ref_starts, ref_ends = [], []
    for b in batch:
        s, e = sample_reference_crop(b["T_ae"], kc=kc, rng=rng)
        ref_starts.append(s); ref_ends.append(e)
    T_ref_max = max(e - s for s, e in zip(ref_starts, ref_ends))

    text_ids       = torch.zeros(B, T_text_max, dtype=torch.long)
    text_mask      = torch.zeros(B, 1, T_text_max, dtype=torch.float32)
    z_ttl          = torch.zeros(B, 144, L_ttl_max, dtype=torch.float32)
    latent_mask    = torch.zeros(B, 1, L_ttl_max, dtype=torch.float32)
    ref_loss_mask  = torch.zeros(B, 1, L_ttl_max, dtype=torch.float32)
    z_ae_ref       = torch.zeros(B, 24, T_ref_max, dtype=torch.float32)
    ref_frame_mask = torch.zeros(B, 1, T_ref_max, dtype=torch.float32)

    for i, b in enumerate(batch):
        t_len = b["text_len"]
        text_ids [i, :t_len] = b["text_ids"]
        text_mask[i, 0, :t_len] = 1.0

        z_ae_i = b["z_ae"]    # [24, T_i]
        T_i = z_ae_i.shape[-1]

        # full TTL latent (target)
        z_ttl_i = prepare_ttl_latent(z_ae_i, mean, std, scale=scale, kc=kc)  # [144, L_i]
        L_i = z_ttl_i.shape[-1]
        z_ttl[i, :, :L_i] = z_ttl_i
        latent_mask[i, 0, :L_i] = 1.0

        # ref crop in AE space
        s, e = ref_starts[i], ref_ends[i]
        z_ae_crop = z_ae_i[:, s:e]
        T_crop = z_ae_crop.shape[-1]
        # z-score normalize the crop (single normalization, paper-faithful)
        z_ae_norm = (z_ae_crop - mean.view(-1, 1)) / std.view(-1, 1)
        z_ae_ref[i, :, :T_crop] = z_ae_norm
        ref_frame_mask[i, 0, :T_crop] = 1.0

        # ref loss mask in TTL-frame space (m: 1 outside crop, 0 inside)
        # AE frame [s, e) → TTL frame [s/kc, e/kc) since we ensured s, e are kc-aligned
        s_ttl = s // kc
        e_ttl = e // kc
        # default: 1 on valid (non-pad), 0 on pad
        rmask_i = torch.zeros(L_i)
        rmask_i[:s_ttl] = 1.0
        rmask_i[e_ttl:L_i] = 1.0
        ref_loss_mask[i, 0, :L_i] = rmask_i

    return {
        "z_ttl": z_ttl,
        "latent_mask": latent_mask,
        "ref_loss_mask": ref_loss_mask,
        "z_ae_ref": z_ae_ref,
        "ref_frame_mask": ref_frame_mask,
        "text_ids": text_ids,
        "text_mask": text_mask,
    }


def sample_dp_reference_crop(T_ae: int, rng: random.Random | None = None) -> tuple[int, int]:
    """DP reference crop sampler (paper Sec 4.2 line 372):
       'randomly selecting a segment from 5% to 95% of the input speech.'
    Crop LENGTH ~ U[0.05*T, 0.95*T], start position uniform in [0, T-len].
    Returns (start, end) inclusive-exclusive AE-frame indices.
    """
    rng = rng or random
    min_len = max(1, int(round(0.05 * T_ae)))
    max_len = max(min_len, int(round(0.95 * T_ae)))
    crop_len = rng.randint(min_len, max_len) if max_len > min_len else min_len
    start_max = T_ae - crop_len
    start = rng.randint(0, start_max) if start_max > 0 else 0
    return start, start + crop_len


SR_DP = 44100
HOP_DP = 512


def collate_dp(batch: list[dict], mean: torch.Tensor, std: torch.Tensor,
               scale: float = TTL_NORMALIZER_SCALE,
               rng: random.Random | None = None):
    """DP collate (paper Sec 4.2):
      - reference encoder receives a 5-95% random crop of the FULL AE latent
      - target = total utterance duration in seconds
      - latent z-scored once (paper-faithful, no extra scale)

    Returns:
        text_ids           : [B, T_text_max] long
        text_mask          : [B, 1, T_text_max]
        z_ae_ref_dp        : [B, 24, T_ref_max]  cropped + normalized AE latent
        ref_dp_frame_mask  : [B, 1, T_ref_max]
        gt_duration_sec    : [B]    target (seconds)
    """
    rng = rng or random
    B = len(batch)
    T_text_max = max(b["text_len"] for b in batch)

    ref_starts, ref_ends = [], []
    for b in batch:
        s, e = sample_dp_reference_crop(b["T_ae"], rng=rng)
        ref_starts.append(s); ref_ends.append(e)
    T_ref_max = max(e - s for s, e in zip(ref_starts, ref_ends))

    text_ids          = torch.zeros(B, T_text_max, dtype=torch.long)
    text_mask         = torch.zeros(B, 1, T_text_max, dtype=torch.float32)
    z_ae_ref_dp       = torch.zeros(B, 24, T_ref_max, dtype=torch.float32)
    ref_dp_frame_mask = torch.zeros(B, 1, T_ref_max, dtype=torch.float32)
    gt_duration_sec   = torch.zeros(B, dtype=torch.float32)

    for i, b in enumerate(batch):
        t_len = b["text_len"]
        text_ids [i, :t_len] = b["text_ids"]
        text_mask[i, 0, :t_len] = 1.0

        z_ae_i = b["z_ae"]
        T_i = z_ae_i.shape[-1]
        s, e = ref_starts[i], ref_ends[i]
        z_crop = z_ae_i[:, s:e]
        z_norm = (z_crop - mean.view(-1, 1)) / std.view(-1, 1) * scale
        T_crop = z_norm.shape[-1]
        z_ae_ref_dp[i, :, :T_crop] = z_norm
        ref_dp_frame_mask[i, 0, :T_crop] = 1.0

        gt_duration_sec[i] = T_i * HOP_DP / SR_DP

    return {
        "text_ids": text_ids,
        "text_mask": text_mask,
        "z_ae_ref_dp": z_ae_ref_dp,
        "ref_dp_frame_mask": ref_dp_frame_mask,
        "gt_duration_sec": gt_duration_sec,
    }


if __name__ == "__main__":
    print("This module expects a filled cache_dir. Run `cache_latents.py` first, "
          "then try:\n"
          "  ds = TTLDataset('training/runs/ae_v1/cache', 'assets/onnx/unicode_indexer.json')\n"
          "  item = ds[0]")
    # Smoke test sample_reference_crop
    rng = random.Random(0)
    for T_ae in [40, 200, 1000]:
        for _ in range(3):
            s, e = sample_reference_crop(T_ae, kc=6, rng=rng)
            print(f"  T_ae={T_ae}: ref crop [{s}, {e}) len={e-s} ({(e-s)*512/44100:.2f}s)")
