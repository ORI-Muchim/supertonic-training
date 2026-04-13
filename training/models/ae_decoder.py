"""Speech Autoencoder Decoder (training-time, pure).

The shipped `vocoder.onnx` is the same decoder but wrapped with TTL-specific
pre-processing (un-chunking 6× + latent de-normalization via latent_mean/std).
During AE training we operate purely in AE latent space, so those wrappers are
handled externally. This module is the inverse of `AEEncoder`.

Structure (from assets/onnx/tts.json > ae.decoder, verified in
analysis/ARCHITECTURE_MAP.md and analysis/torch_vocoder.py):

    latent [B, 24, T_frames]
      → causal Conv1d 24 → 512  (ksz=7)                        # stem
      → 10 × ConvNeXt (ksz=7, dilations [1,2,4,1,2,4,1,1,1,1]) # main stack
      → BatchNorm1d(512)                                       # final_norm
      → causal Conv1d 512 → 2048 (ksz=3)                       # head.layer1
      → PReLU (1 shared channel)
      → Conv1d 2048 → 512 (ksz=1, no bias)                     # head.layer2
      → transpose [0,2,1] + flatten → waveform [B, T_frames * 512]

The `512` in head output is `hop_length`: each frame produces 512 output samples.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .common import ConvNeXt1D, causal_pad, symmetric_pad


class AEDecoder(nn.Module):
    def __init__(
        self,
        ldim: int = 24,
        hdim: int = 512,
        intermediate_dim: int = 2048,
        ksz_init: int = 7,
        ksz: int = 7,
        num_layers: int = 10,
        dilation_lst: tuple[int, ...] = (1, 2, 4, 1, 2, 4, 1, 1, 1, 1),
        head_inter: int = 2048,
        head_out: int = 512,  # = hop_length
        head_ksz: int = 3,
        pad_mode: str = "causal",
    ):
        super().__init__()
        assert len(dilation_lst) == num_layers
        self.ksz_init = ksz_init
        self.head_ksz = head_ksz
        self.head_out = head_out  # hop_length (samples per frame)
        self.pad_mode = pad_mode

        self.stem = nn.Conv1d(ldim, hdim, ksz_init, bias=True)
        self.convnext = nn.ModuleList([
            ConvNeXt1D(hdim, intermediate_dim, ksz, d, pad_mode=pad_mode)
            for d in dilation_lst
        ])
        self.final_norm = nn.BatchNorm1d(hdim)
        self.head_layer1 = nn.Conv1d(hdim, head_inter, head_ksz, bias=True)
        self.head_act = nn.PReLU(num_parameters=1)
        self.head_layer2 = nn.Conv1d(head_inter, head_out, 1, bias=False)

    def _pad(self, x, ksz, d=1):
        return causal_pad(x, ksz, d) if self.pad_mode == "causal" else symmetric_pad(x, ksz, d)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z [B, ldim, T_frames] -> wav [B, T_frames * hop_length]."""
        B, _, T = z.shape
        x = self.stem(self._pad(z, self.ksz_init))
        for blk in self.convnext:
            x = blk(x)
        x = self.final_norm(x)
        x = self.head_layer1(self._pad(x, self.head_ksz))
        x = self.head_act(x)
        x = self.head_layer2(x)                               # [B, head_out, T]
        wav = x.transpose(1, 2).reshape(B, T * self.head_out) # [B, T*hop]
        return wav

    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    dec = AEDecoder()
    print(f"AE decoder params: {dec.num_params():,}")
    z = torch.randn(2, 24, 87)
    wav = dec(z)
    print(f"latent {tuple(z.shape)} -> wav {tuple(wav.shape)}")
    assert wav.shape == (2, 87 * 512), f"expected (2, {87*512}), got {wav.shape}"
    print("OK: shapes verified.")
