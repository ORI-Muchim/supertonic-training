"""Losses for AE (Stage 1) training.

Paper (SupertonicTTS, arXiv 2503.23108):
  L_G = 45·L_recon + 1·L_adv + 0.1·L_fm
  L_recon: multi-resolution **mel** L1 at FFT ∈ {1024, 2048, 4096}
  L_adv, L_D: LSGAN style (MSE on labels 0/1) — HiFi-GAN convention
  L_fm:     L1 feature matching across every discriminator layer
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class MultiResolutionMelLoss(nn.Module):
    """L1 between multi-resolution log-mel spectrograms of real vs fake waveform."""
    def __init__(
        self,
        sample_rate: int = 44100,
        ffts: tuple[int, ...] = (1024, 2048, 4096),
        n_mels_per_fft: tuple[int, ...] | None = None,     # one per FFT size
        eps: float = 1e-5,
    ):
        super().__init__()
        self.eps = eps
        # Scale n_mels with FFT size to avoid empty filterbanks at small FFT.
        # Default: 80, 160, 228 — covers common BigVGAN/DAC-style configs and
        # reaches the AE input resolution (228) at the largest FFT.
        if n_mels_per_fft is None:
            # Chosen so no filterbank is empty at SR=44.1kHz:
            # FFT 1024 -> 80, FFT 2048 -> 160, FFT 4096 -> 228.
            _default_map = {1024: 80, 2048: 160, 4096: 228, 8192: 228}
            n_mels_per_fft = tuple(_default_map.get(f, max(80, min(228, f // 16))) for f in ffts)
        assert len(n_mels_per_fft) == len(ffts)
        self.mel_transforms = nn.ModuleList()
        for n_fft, n_mels in zip(ffts, n_mels_per_fft):
            self.mel_transforms.append(torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=n_fft,
                hop_length=n_fft // 4,
                n_mels=n_mels,
                f_min=0.0, f_max=sample_rate / 2,
                power=1.0,
                center=True, pad_mode="reflect",
            ))

    def forward(self, wav_hat: torch.Tensor, wav: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for mel in self.mel_transforms:
            # Use clamp inside log (not + eps) to prevent gradient explosion:
            # d/dx log(x + eps) = 1/(x+eps) -> 1e5 when x ~ 0, blowing up backward.
            # clamp_min flattens grad below eps, which is fine — near-silence bins
            # shouldn't dominate training anyway.
            m_hat = torch.log(mel(wav_hat).clamp_min(self.eps))
            m_ref = torch.log(mel(wav).clamp_min(self.eps))
            loss = loss + F.l1_loss(m_hat, m_ref)
        return loss / len(self.mel_transforms)


# -------------------- GAN + FM losses --------------------
def generator_adv_loss(fake_logits_list: list[torch.Tensor]) -> torch.Tensor:
    """LSGAN generator loss: E[(D(fake) - 1)^2] averaged over heads."""
    loss = 0.0
    for lg in fake_logits_list:
        loss = loss + torch.mean((lg - 1.0) ** 2)
    return loss / len(fake_logits_list)


def discriminator_adv_loss(
    real_logits_list: list[torch.Tensor],
    fake_logits_list: list[torch.Tensor],
) -> torch.Tensor:
    """LSGAN discriminator loss."""
    loss = 0.0
    for lr, lf in zip(real_logits_list, fake_logits_list):
        loss = loss + torch.mean((lr - 1.0) ** 2) + torch.mean(lf ** 2)
    return loss / len(real_logits_list)


def feature_matching_loss(
    real_feats_per_head: list[list[torch.Tensor]],
    fake_feats_per_head: list[list[torch.Tensor]],
) -> torch.Tensor:
    """Sum of L1 distances between real and fake discriminator features,
    normalized by total number of feature maps."""
    total = 0.0
    n = 0
    for rf_list, ff_list in zip(real_feats_per_head, fake_feats_per_head):
        for rf, ff in zip(rf_list, ff_list):
            total = total + F.l1_loss(ff, rf.detach())
            n += 1
    return total / max(n, 1)


if __name__ == "__main__":
    # Smoke test
    torch.manual_seed(0)
    wav1 = torch.randn(2, 44100)
    wav2 = wav1 + 0.1 * torch.randn_like(wav1)
    mel_loss = MultiResolutionMelLoss()
    print(f"mel loss: {mel_loss(wav2, wav1).item():.4f}")
    # GAN test
    from training.models.discriminators import AEDiscriminator
    D = AEDiscriminator()
    lr, fr = D(wav1)
    lf, ff = D(wav2.detach())
    print(f"L_adv_G: {generator_adv_loss(lf).item():.4f}")
    print(f"L_D:     {discriminator_adv_loss(lr, lf).item():.4f}")
    print(f"L_fm:    {feature_matching_loss(fr, ff).item():.4f}")
