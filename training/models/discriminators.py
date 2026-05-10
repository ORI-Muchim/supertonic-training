"""Discriminators for AE GAN training (Stage 1).

Paper-faithful (SupertonicTTS arXiv 2503.23108, App. A.1.3 + Table 7):

  MPD (Multi-Period Discriminator): folds wav into 2D by periodic slicing.
    Periods: 2, 3, 5, 7, 11.
    Per-disc: 6 conv layers, output channels [16, 64, 256, 512, 512, 1].

  MRD (Multi-Resolution Discriminator): operates on |STFT| magnitude.
    FFT sizes: 512, 1024, 2048. Hops = FFT/4. Window = Hann.
    Per-disc: 6 Conv2D layers (Table 7), all 16-channel hidden, kernel (5,5)
    except the last (3,3). Strides only on frequency axis: (1,1),(2,1),(2,1),(2,1),(1,1),(1,1).

Both return (logits, feature_list) where feature_list holds every intermediate
activation — used by the feature-matching loss.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ========================= MPD =========================
class PeriodDiscriminator(nn.Module):
    """Paper App. A.1.3: 6 conv layers, hidden output channels [16,64,256,512,512,1]."""
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        self.kernel_size = kernel_size
        # Layer outputs per paper: 16, 64, 256, 512, 512, 1.
        # We split into 5 hidden convs + 1 final post-conv to keep feature-matching
        # taps on hidden activations (LeakyReLU outputs).
        ch = [1, 16, 64, 256, 512, 512]      # input + 5 hidden
        self.convs = nn.ModuleList()
        for i in range(len(ch) - 1):
            # Stride on the time axis for downsampling, except the last hidden which stays at 1.
            self.convs.append(weight_norm(nn.Conv2d(
                ch[i], ch[i+1],
                kernel_size=(kernel_size, 1),
                stride=(stride if i < len(ch) - 2 else 1, 1),
                padding=((kernel_size - 1) // 2, 0),
            )))
        # 6th layer: collapse to logits.
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
    """STFT magnitude discriminator (paper Table 7).

    6 Conv2D layers on log-|STFT| spectrogram. All hidden 16 channels.
    Strides only on frequency axis (paper specifies stride=(2,1) for
    layers 2-4, (1,1) for 1 / 5 / 6). Kernels (5,5) except final (3,3).
    """
    def __init__(self, n_fft: int, hop_length: int, win_length: int, channels: int = 16):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
        # Table 7 schedule. (in, out, kernel, stride):
        layer_specs = [
            (1,        channels, (5, 5), (1, 1)),   # layer 1
            (channels, channels, (5, 5), (2, 1)),   # layer 2
            (channels, channels, (5, 5), (2, 1)),   # layer 3
            (channels, channels, (5, 5), (2, 1)),   # layer 4
            (channels, channels, (5, 5), (1, 1)),   # layer 5
        ]
        self.convs = nn.ModuleList()
        for c_in, c_out, ksz, stride in layer_specs:
            pad = (ksz[0] // 2, ksz[1] // 2)
            self.convs.append(weight_norm(nn.Conv2d(c_in, c_out, ksz, stride, padding=pad)))
        # Layer 6: collapse to logits, kernel (3,3), stride (1,1).
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1)))

    def _spec(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, T] -> log-magnitude spectrogram [B, 1, F, T] (paper input)."""
        spec = torch.stft(
            wav, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window,
            center=True, pad_mode="reflect", return_complex=True,
        )
        log_mag = torch.log(spec.abs().clamp_min(1e-5))   # log-scaled per paper App A.1.3
        return log_mag.unsqueeze(1)

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
        channels: int = 16,
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
