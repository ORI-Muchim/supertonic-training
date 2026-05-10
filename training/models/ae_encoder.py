"""Speech Autoencoder Encoder.

Mirrors the AE decoder structure (found in assets/onnx/vocoder.onnx, see
analysis/ARCHITECTURE_MAP.md for reverse-engineered details).

Config from assets/onnx/tts.json > ae.encoder:
    ksz_init: 7, ksz: 7, num_layers: 10
    dilation_lst: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]    (encoder: all 1)
    intermediate_dim: 2048, idim: 1253, hdim: 512, odim: 24

Input:
    features [B, 1253, T_frames] from SpecProcessor
      (concat of log-STFT-magnitude 1025 + log-mel 228)
Output:
    latent   [B, 24, T_frames]
      (AE latent, temporally aligned 1:1 with spectrogram frames.
       hop=512 at 44.1 kHz → latent frame rate ≈ 86.13 Hz.)

Padding: paper specifies causality only for the **decoder** (Sec 3.1.1, A.1.2);
the encoder is non-causal → symmetric replicate padding.
Norms (paper A.1.1): BatchNorm after stem, LayerNorm after final projection.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .common import ConvNeXt1D, causal_pad, symmetric_pad


class AEEncoder(nn.Module):
    def __init__(
        self,
        idim: int = 228,
        hdim: int = 512,
        odim: int = 24,
        ksz_init: int = 7,
        ksz: int = 7,
        num_layers: int = 10,
        intermediate_dim: int = 2048,
        dilation_lst: tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        pad_mode: str = "symmetric",
        enable_stem_bn: bool = True,
        enable_out_ln: bool = True,
    ):
        super().__init__()
        assert len(dilation_lst) == num_layers, "dilation_lst length must equal num_layers"
        self.ksz_init = ksz_init
        self.pad_mode = pad_mode
        self.enable_stem_bn = enable_stem_bn
        self.enable_out_ln = enable_out_ln

        self.stem = nn.Conv1d(idim, hdim, ksz_init, bias=True)
        self.stem_norm = nn.BatchNorm1d(hdim) if enable_stem_bn else nn.Identity()
        self.convnext = nn.ModuleList([
            ConvNeXt1D(hdim, intermediate_dim, ksz, d, pad_mode=pad_mode)
            for d in dilation_lst
        ])
        self.proj_out = nn.Conv1d(hdim, odim, 1, bias=False)
        self.out_norm = nn.LayerNorm(odim, eps=1e-6) if enable_out_ln else nn.Identity()

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats [B, idim, T] -> latent [B, odim, T]."""
        if self.pad_mode == "causal":
            x = causal_pad(feats, self.ksz_init, 1)
        else:
            x = symmetric_pad(feats, self.ksz_init, 1)
        x = self.stem(x)
        x = self.stem_norm(x)
        for blk in self.convnext:
            x = blk(x)
        z = self.proj_out(x)
        if self.enable_out_ln:
            z = self.out_norm(z.transpose(1, 2)).transpose(1, 2)
        return z

    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    enc = AEEncoder()
    print(f"AE encoder params: {enc.num_params():,}")
    # Dummy forward: 1 sec audio → 87 frames (paper-faithful: 228-dim mel input)
    feats = torch.randn(2, 228, 87)
    z = enc(feats)
    print(f"feats {tuple(feats.shape)} -> latent {tuple(z.shape)}")
    assert z.shape == (2, 24, 87)
    print("OK: shapes verified.")
