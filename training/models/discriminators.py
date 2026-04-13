"""Discriminators for AE GAN training (Stage 1).

Two discriminators, each following HiFi-GAN conventions:

  MPD (Multi-Period Discriminator): folds wav into 2D by periodic slicing.
    Periods: 2, 3, 5, 7, 11.  (from SupertonicTTS paper)

  MRD (Multi-Resolution Discriminator): operates on STFT magnitude spectrograms.
    FFT sizes: 512, 1024, 2048.   (from paper)

Both return (logits, feature_list) where feature_list contains every intermediate
activation — used by the feature-matching loss.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ========================= MPD =========================
class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        self.kernel_size = kernel_size
        ch = [1, 32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList()
        for i in range(len(ch) - 1):
            self.convs.append(weight_norm(nn.Conv2d(
                ch[i], ch[i+1],
                kernel_size=(kernel_size, 1),
                stride=(stride if i < len(ch) - 2 else 1, 1),
                padding=((kernel_size - 1) // 2, 0),
            )))
        self.conv_post = weight_norm(nn.Conv2d(ch[-1], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, wav: torch.Tensor):
        """wav [B, T] -> (logits [B, *], features [list of tensors])."""
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)   # [B, 1, T]
        B, C, T = wav.shape
        # pad to multiple of period
        rem = T % self.period
        if rem != 0:
            wav = F.pad(wav, (0, self.period - rem), "reflect")
            T = T + (self.period - rem)
        x = wav.view(B, C, T // self.period, self.period)  # [B, 1, T/P, P]

        feats = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feats.append(x)
        x = self.conv_post(x)
        feats.append(x)
        logits = x.flatten(1)  # [B, *]
        return logits, feats


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discs = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, wav: torch.Tensor):
        """wav [B, T] -> (logits_list, features_list_of_lists)."""
        logits, feats = [], []
        for d in self.discs:
            lg, ft = d(wav)
            logits.append(lg)
            feats.append(ft)
        return logits, feats


# ========================= MRD =========================
class ResolutionDiscriminator(nn.Module):
    """STFT magnitude discriminator at one (n_fft, hop, win) resolution."""
    def __init__(self, n_fft: int, hop_length: int, win_length: int, channels: int = 32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
        # 2D conv stack on [B, 1, F, T] spectrogram magnitude
        ch = [1, channels, channels, channels, channels]
        self.convs = nn.ModuleList()
        for i in range(len(ch) - 1):
            stride = (2, 2) if i > 0 else (1, 1)
            self.convs.append(weight_norm(nn.Conv2d(
                ch[i], ch[i+1],
                kernel_size=(3, 9),
                stride=stride,
                padding=(1, 4),
            )))
        self.conv_post = weight_norm(nn.Conv2d(ch[-1], 1, (3, 3), 1, padding=(1, 1)))

    def _spec(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, T] -> magnitude spectrogram [B, 1, F, T]."""
        spec = torch.stft(
            wav, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window,
            center=True, pad_mode="reflect", return_complex=True,
        )
        mag = spec.abs()
        return mag.unsqueeze(1)

    def forward(self, wav: torch.Tensor):
        x = self._spec(wav)
        feats = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feats.append(x)
        x = self.conv_post(x)
        feats.append(x)
        logits = x.flatten(1)
        return logits, feats


class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        ffts: tuple[int, ...] = (512, 1024, 2048),
        hops: tuple[int, ...] | None = None,
        wins: tuple[int, ...] | None = None,
        channels: int = 32,
    ):
        super().__init__()
        hops = hops or tuple(f // 4 for f in ffts)
        wins = wins or ffts
        assert len(ffts) == len(hops) == len(wins)
        self.discs = nn.ModuleList([
            ResolutionDiscriminator(n, h, w, channels) for n, h, w in zip(ffts, hops, wins)
        ])

    def forward(self, wav: torch.Tensor):
        logits, feats = [], []
        for d in self.discs:
            lg, ft = d(wav)
            logits.append(lg)
            feats.append(ft)
        return logits, feats


# ========================= combined =========================
class AEDiscriminator(nn.Module):
    """Combines MPD + MRD. Returns (logits, features) as concatenated lists."""
    def __init__(
        self,
        mpd_periods: tuple[int, ...] = (2, 3, 5, 7, 11),
        mrd_ffts: tuple[int, ...] = (512, 1024, 2048),
    ):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(mpd_periods)
        self.mrd = MultiResolutionDiscriminator(mrd_ffts)

    def forward(self, wav: torch.Tensor):
        mp_l, mp_f = self.mpd(wav)
        mr_l, mr_f = self.mrd(wav)
        return mp_l + mr_l, mp_f + mr_f


if __name__ == "__main__":
    D = AEDiscriminator()
    n_params = sum(p.numel() for p in D.parameters())
    print(f"AE discriminator params: {n_params:,} (~{n_params/1e6:.1f}M)")
    wav = torch.randn(2, 44100)
    logits, feats = D(wav)
    print(f"num disc heads: {len(logits)} (MPD 5 + MRD 3 = 8 expected)")
    for i, (lg, ft) in enumerate(zip(logits, feats)):
        print(f"  head {i}: logits {tuple(lg.shape)}, feats {len(ft)} levels, last {tuple(ft[-1].shape)}")
