"""Losses for AE (Stage 1) training.

Paper (SupertonicTTS, arXiv 2503.23108):
  L_G = 45·L_recon + 1·L_adv + 0.1·L_fm
  L_recon: multi-resolution **mel** L1 at FFT ∈ {1024, 2048, 4096}
  L_adv, L_D: LSGAN with ±1 labels (paper Eq. 4/5), NOT 0/1 HiFi-GAN labels
  L_fm:     L1 feature matching across every discriminator layer
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class MultiResolutionMelLoss(nn.Module):
    """L1 between multi-resolution log-mel spectrograms of real vs fake waveform.

    Paper (App. B.1): FFT sizes 1024 / 2048 / 4096, mel bands 64 / 128 / 128,
    hops = FFT / 4, Hann windows of size = FFT.
    """
    def __init__(
        self,
        sample_rate: int = 44100,
        ffts: tuple[int, ...] = (1024, 2048, 4096),
        n_mels_per_fft: tuple[int, ...] | None = None,     # one per FFT size
        eps: float = 1e-5,
        reduction: str = "mean",  # "mean" keeps lambda_recon scale independent of #resolutions.
    ):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        # Paper App. B.1: 64 / 128 / 128 mel bands for FFT 1024 / 2048 / 4096.
        if n_mels_per_fft is None:
            _default_map = {1024: 64, 2048: 128, 4096: 128, 8192: 128}
            n_mels_per_fft = tuple(_default_map.get(f, max(64, min(128, f // 16))) for f in ffts)
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
        # The paper defines a multi-resolution mel spectral L1 loss but does not
        # specify whether resolution terms are summed or averaged. Use "mean" by
        # default so lambda_recon=45 is not implicitly multiplied by len(ffts).
        loss = 0.0
        for mel in self.mel_transforms:
            m_hat = torch.log(mel(wav_hat).clamp_min(self.eps))
            m_ref = torch.log(mel(wav).clamp_min(self.eps))
            loss = loss + F.l1_loss(m_hat, m_ref)
        if self.reduction == "mean":
            loss = loss / len(self.mel_transforms)
        return loss


# -------------------- GAN + FM losses --------------------
# Paper Eq. (4): L_adv(G;D) = E[(D(G(x)) - 1)^2]      — G wants D(fake) -> +1
# Paper Eq. (5): L_adv(D;G) = E[(D(G(x)) + 1)^2 + (D(x) - 1)^2]
#                                                     — D wants D(real) -> +1, D(fake) -> -1
# This is the LS-GAN with -1/+1 labels (HiFi-GAN evolved variant), NOT 0/1 labels.
def generator_adv_loss(fake_logits_list: list[torch.Tensor]) -> torch.Tensor:
    """LSGAN-(±1) generator loss: E[(D(fake) - 1)^2] averaged over heads (paper Eq. 4)."""
    loss = 0.0
    for lg in fake_logits_list:
        loss = loss + torch.mean((lg - 1.0) ** 2)
    return loss / len(fake_logits_list)


def discriminator_adv_loss(
    real_logits_list: list[torch.Tensor],
    fake_logits_list: list[torch.Tensor],
) -> torch.Tensor:
    """LSGAN-(±1) discriminator loss (paper Eq. 5)."""
    loss = 0.0
    for lr, lf in zip(real_logits_list, fake_logits_list):
        loss = loss + torch.mean((lr - 1.0) ** 2) + torch.mean((lf + 1.0) ** 2)
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
