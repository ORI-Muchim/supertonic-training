"""Dataset for TTL (Stage 2) training — loads cached AE latents + text_ids.

Depends on `cache_latents.py` having been run first.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "py"))
from helper import UnicodeProcessor  # type: ignore


# normalizer.scale from tts.json (TTL latent space scale)
TTL_NORMALIZER_SCALE = 0.25


def prepare_ttl_latent(
    z_ae: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    scale: float = TTL_NORMALIZER_SCALE,
    kc: int = 6,
) -> torch.Tensor:
    """Convert raw AE latent [C, T] → normalized chunk-compressed TTL latent [C*kc, T/kc].

    z_ttl = chunk_compress((z_ae - mean) / std * scale)
    At inference, vocoder inverts this via un-chunk + * std + mean / scale.
    """
    mean = mean.to(z_ae.device).view(-1, 1)
    std  = std.to(z_ae.device).view(-1, 1)
    z_norm = (z_ae - mean) / std * scale
    # chunk_compress: pad T to multiple of kc (replicate), then fold kc into channels
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


class TTLDataset(Dataset):
    """Returns one utterance per __getitem__ (variable length — collate with padding).

    Item dict:
        text_ids   : LongTensor [T_text]
        text_len   : int
        z_ae       : FloatTensor [24, T_ae]  (raw AE latent, to be normalized in collate)
        lang       : str  (default "ko" for KSS)
        text_raw   : original text (for debugging / logging)
    """
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
        # Tokenize text
        text_ids_np, _text_mask_np = self.text_processor([m["text_norm"]], [self.lang])
        # text_ids_np shape [1, T_text] → squeeze
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
                kc: int = 6, scale: float = TTL_NORMALIZER_SCALE):
    """Collate variable-length items into a padded batch.

    Returns a dict with:
        z_ttl        : [B, 144, L_max]         chunk-compressed, normalized (target for FM)
        latent_mask  : [B, 1, L_max]           TTL-frame mask
        z_ae         : [B, 24, T_ae_max]       raw AE latent (padded) — input to style encoder
        frame_mask   : [B, 1, T_ae_max]        AE-frame mask (for style encoder attention masking)
        text_ids     : [B, T_max]
        text_mask    : [B, 1, T_max]
    """
    B = len(batch)
    T_text_max = max(b["text_len"] for b in batch)
    # latent length in TTL (chunk-compressed) space
    L_ttl_max = max((b["T_ae"] + kc - 1) // kc for b in batch)
    # AE frame length, padded to multiple of kc
    T_ae_max = L_ttl_max * kc

    text_ids   = torch.zeros(B, T_text_max, dtype=torch.long)
    text_mask  = torch.zeros(B, 1, T_text_max, dtype=torch.float32)
    z_ttl      = torch.zeros(B, 144, L_ttl_max, dtype=torch.float32)
    lat_mask   = torch.zeros(B, 1, L_ttl_max, dtype=torch.float32)
    z_ae_pad   = torch.zeros(B, 24, T_ae_max, dtype=torch.float32)
    frame_mask = torch.zeros(B, 1, T_ae_max, dtype=torch.float32)

    for i, b in enumerate(batch):
        t_len = b["text_len"]
        text_ids [i, :t_len] = b["text_ids"]
        text_mask[i, 0, :t_len] = 1.0

        z_ae_i = b["z_ae"]    # [24, T_i]
        T_i = z_ae_i.shape[-1]
        z_ae_pad[i, :, :T_i] = z_ae_i
        frame_mask[i, 0, :T_i] = 1.0

        z_ttl_i = prepare_ttl_latent(z_ae_i, mean, std, scale=scale, kc=kc)  # [144, L_i]
        L_i = z_ttl_i.shape[-1]
        z_ttl [i, :, :L_i] = z_ttl_i
        lat_mask[i, 0, :L_i] = 1.0

    return {
        "z_ttl": z_ttl,
        "latent_mask": lat_mask,
        "z_ae": z_ae_pad,
        "frame_mask": frame_mask,
        "text_ids": text_ids,
        "text_mask": text_mask,
    }


if __name__ == "__main__":
    print("This module expects a filled cache_dir. Run `cache_latents.py` first, "
          "then try:\n"
          "  ds = TTLDataset('training/runs/ae_v1/cache', 'assets/onnx/unicode_indexer.json')\n"
          "  item = ds[0]")
