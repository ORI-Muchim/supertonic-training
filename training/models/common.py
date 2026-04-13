"""Shared building blocks used across Supertonic models.

ConvNeXt1D with causal or symmetric replicate padding.
Reused from analysis/torch_vocoder.py but reorganized for training use (optional mask).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_pad(x: torch.Tensor, ksz: int, dilation: int = 1) -> torch.Tensor:
    """Left pad by (ksz-1)*dilation with replicate mode."""
    return F.pad(x, ((ksz - 1) * dilation, 0), mode="replicate")


def symmetric_pad(x: torch.Tensor, ksz: int, dilation: int = 1) -> torch.Tensor:
    """Symmetric replicate pad by (ksz-1)*dilation/2 on each side (ksz must be odd)."""
    pad = (ksz - 1) * dilation // 2
    return F.pad(x, (pad, pad), mode="replicate")


class ConvNeXt1D(nn.Module):
    """ConvNeXt block in 1D, NCL layout throughout.

    DWConv → LayerNorm (channels-last) → PWConv up → GELU → PWConv down → γ · → + residual.

    `pad_mode` ∈ {"causal", "symmetric"}. All Supertonic ConvNeXts use "replicate" fill.

    If `mask` is provided in forward, it's applied after the residual sum (shape [B, 1, T]).
    Mask before/after each conv is NOT done here; for training AE we don't need masking since
    we batch same-length clips. For text/latent models that handle variable lengths, use
    a different class (see analysis/torch_*.py).
    """
    def __init__(
        self,
        dim: int,
        intermediate: int,
        ksz: int,
        dilation: int = 1,
        pad_mode: str = "causal",
    ):
        super().__init__()
        self.ksz = ksz
        self.dilation = dilation
        self.pad_mode = pad_mode
        self.dwconv = nn.Conv1d(dim, dim, ksz, groups=dim, dilation=dilation, bias=True)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv1d(dim, intermediate, 1, bias=True)
        self.pwconv2 = nn.Conv1d(intermediate, dim, 1, bias=True)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))

    def _pad(self, x):
        if self.pad_mode == "causal":
            return causal_pad(x, self.ksz, self.dilation)
        elif self.pad_mode == "symmetric":
            return symmetric_pad(x, self.ksz, self.dilation)
        raise ValueError(f"unknown pad_mode: {self.pad_mode}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.dwconv(self._pad(x))
        # LN on channel dim: NCL -> NLC -> NCL
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.pwconv1(x)
        x = F.gelu(x, approximate="none")
        x = self.pwconv2(x)
        x = self.gamma * x
        out = residual + x
        if mask is not None:
            out = out * mask
        return out
