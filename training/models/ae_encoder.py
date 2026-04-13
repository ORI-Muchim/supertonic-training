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

Padding: causal replicate (mirrors decoder for streaming-friendly alignment).
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .common import ConvNeXt1D, causal_pad


class AEEncoder(nn.Module):
    def __init__(
        self,
        idim: int = 1253,
        hdim: int = 512,
        odim: int = 24,
        ksz_init: int = 7,
        ksz: int = 7,
        num_layers: int = 10,
        intermediate_dim: int = 2048,
        dilation_lst: tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        pad_mode: str = "causal",
    ):
        super().__init__()
        assert len(dilation_lst) == num_layers, "dilation_lst length must equal num_layers"
        self.ksz_init = ksz_init
        self.pad_mode = pad_mode

        self.stem = nn.Conv1d(idim, hdim, ksz_init, bias=True)
        self.convnext = nn.ModuleList([
            ConvNeXt1D(hdim, intermediate_dim, ksz, d, pad_mode=pad_mode)
            for d in dilation_lst
        ])
        # Final 1×1 projection to latent dim. No bias to match common autoencoder patterns.
        self.proj_out = nn.Conv1d(hdim, odim, 1, bias=False)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats [B, idim, T] -> latent [B, odim, T]."""
        # Stem
        if self.pad_mode == "causal":
            x = causal_pad(feats, self.ksz_init, 1)
        else:
            from .common import symmetric_pad
            x = symmetric_pad(feats, self.ksz_init, 1)
        x = self.stem(x)
        # 10 × ConvNeXt
        for blk in self.convnext:
            x = blk(x)
        # Final projection
        z = self.proj_out(x)
        return z

    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    enc = AEEncoder()
    print(f"AE encoder params: {enc.num_params():,}")
    # Dummy forward: 1 sec audio → 87 frames
    feats = torch.randn(2, 1253, 87)
    z = enc(feats)
    print(f"feats {tuple(feats.shape)} -> latent {tuple(z.shape)}")
    assert z.shape == (2, 24, 87)
    print("OK: shapes verified.")
