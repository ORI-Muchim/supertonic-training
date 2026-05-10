"""Spectrogram feature extractor for Supertonic AE encoder.

Two configurations:
  - mode="concat"  -> [B, 1253, T_frames] = concat(log_STFT[1025], log_mel[228])
                      Matches shipped ONNX (`tts.json > ae.encoder.idim = 1253`).
                      Use this if downstream code interfaces with shipped vocoder.
  - mode="mel"     -> [B,  228, T_frames] = log_mel[228]
                      Matches the SupertonicTTS paper (Sec 3.1, App B.1):
                      "log-scaled mel spectrogram input ... 228 mel bands".
                      Use this for paper-faithful from-scratch training.

Common parameters (from tts.json > ae.encoder.spec_processor):
  n_fft=2048, win_length=2048, hop_length=512, n_mels=228
  sample_rate=44100, eps=1e-5, norm_mean=0.0, norm_std=1.0
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torchaudio


class SpecProcessor(nn.Module):
    """Convert waveform [B, T] to feature tensor [B, feature_dim, T_frames].

    feature_dim = 1253 (mode="concat") or 228 (mode="mel").
    """
    def __init__(
        self,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        n_mels: int = 228,
        sample_rate: int = 44100,
        eps: float = 1e-5,
        norm_mean: float = 0.0,
        norm_std: float = 1.0,
        mode: str = "concat",
    ):
        super().__init__()
        assert mode in ("concat", "mel"), f"mode must be 'concat' or 'mel', got {mode!r}"
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.eps = eps
        self.mode = mode

        # Spectrogram: outputs |STFT|^2 (power=2) or |STFT| (power=1). We want magnitude.
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            power=1.0, center=True, pad_mode="reflect",
        )
        # Mel filterbank applied to magnitude
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            n_mels=n_mels, power=1.0, center=True, pad_mode="reflect",
            f_min=0.0, f_max=sample_rate / 2,
        )
        self.register_buffer("norm_mean", torch.tensor(norm_mean, dtype=torch.float32))
        self.register_buffer("norm_std",  torch.tensor(norm_std, dtype=torch.float32))

    @property
    def n_stft_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def feature_dim(self) -> int:
        if self.mode == "mel":
            return self.n_mels                       # 228
        return self.n_stft_bins + self.n_mels        # 1025 + 228 = 1253

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, T] -> features [B, feature_dim, T_frames]."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mel = self.mel(wav)                                  # [B, 228, T_f]
        log_mel = torch.log(mel + self.eps)
        if self.mode == "mel":
            feats = log_mel
        else:
            mag = self.spec(wav)                             # [B, 1025, T_f]
            log_mag = torch.log(mag + self.eps)
            feats = torch.cat([log_mag, log_mel], dim=1)     # [B, 1253, T_f]
        feats = (feats - self.norm_mean) / self.norm_std     # (no-op with default 0/1)
        return feats


if __name__ == "__main__":
    # Quick sanity check
    sp = SpecProcessor()
    print(f"feature_dim: {sp.feature_dim}")
    assert sp.feature_dim == 1253, f"expected 1253, got {sp.feature_dim}"

    # 1 second of audio at 44.1 kHz
    wav = torch.randn(2, 44100)
    feats = sp(wav)
    print(f"wav shape: {wav.shape}  ->  feats shape: {feats.shape}")
    # T_frames with center=True, hop=512: ceil(T/hop) + 1 = ceil(44100/512) + 1 = 87 + 1 = 88
    # (actually torchaudio center adds n_fft//2 pad; for T=44100 and hop=512 we get 87 frames)
    assert feats.shape[1] == 1253
    print(f"OK: feature_dim verified = 1253  (= 1025 STFT + 228 mel)")
    print(f"feature stats: min={feats.min():.2f} max={feats.max():.2f} mean={feats.mean():.2f}")
