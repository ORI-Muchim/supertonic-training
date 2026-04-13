"""Speech Autoencoder = SpecProcessor -> Encoder -> (latent) -> Decoder -> waveform."""
from __future__ import annotations
import torch
import torch.nn as nn

from ..data.spectrogram import SpecProcessor
from .ae_encoder import AEEncoder
from .ae_decoder import AEDecoder


class SpeechAutoencoder(nn.Module):
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        n_mels: int = 228,
        ldim: int = 24,
        hdim: int = 512,
        intermediate_dim: int = 2048,
        num_layers: int = 10,
        ksz: int = 7,
        ksz_init: int = 7,
        encoder_dilations: tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        decoder_dilations: tuple[int, ...] = (1, 2, 4, 1, 2, 4, 1, 1, 1, 1),
        pad_mode: str = "causal",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.spec = SpecProcessor(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            n_mels=n_mels, sample_rate=sample_rate,
        )
        idim = self.spec.feature_dim   # 1025 + 228 = 1253
        self.encoder = AEEncoder(
            idim=idim, hdim=hdim, odim=ldim,
            ksz_init=ksz_init, ksz=ksz, num_layers=num_layers,
            intermediate_dim=intermediate_dim,
            dilation_lst=encoder_dilations,
            pad_mode=pad_mode,
        )
        self.decoder = AEDecoder(
            ldim=ldim, hdim=hdim, intermediate_dim=intermediate_dim,
            ksz_init=ksz_init, ksz=ksz, num_layers=num_layers,
            dilation_lst=decoder_dilations,
            head_out=hop_length,
            pad_mode=pad_mode,
        )

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, T_samples] -> latent [B, ldim, T_frames]."""
        feats = self.spec(wav)
        return self.encoder(feats)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """latent [B, ldim, T_frames] -> wav [B, T_frames * hop_length]."""
        return self.decoder(z)

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full AE pass. Returns (wav_reconstructed, latent)."""
        z = self.encode(wav)
        wav_hat = self.decode(z)
        return wav_hat, z

    @torch.no_grad()
    def num_params(self) -> dict:
        return {
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "decoder": sum(p.numel() for p in self.decoder.parameters()),
            "total":   sum(p.numel() for p in self.parameters()),
        }


if __name__ == "__main__":
    import json
    ae = SpeechAutoencoder()
    print("Param counts:", json.dumps(ae.num_params(), indent=2))

    # Sanity check: 1 second of random audio
    wav = torch.randn(2, 44100)
    wav_hat, z = ae(wav)
    print(f"input  wav: {tuple(wav.shape)}")
    print(f"latent    : {tuple(z.shape)}")
    print(f"output wav: {tuple(wav_hat.shape)}")

    # Check that output length is aligned: T_frames * hop_length
    # T_frames = ceil((T_samples + n_fft) / hop) with center=True  → for 44100: 87 frames
    expected_out = 87 * 512
    assert wav_hat.shape[1] == expected_out, f"expected {expected_out}, got {wav_hat.shape[1]}"
    # Note: output length (44544) > input length (44100) due to STFT center padding.
    # In training, we either crop to min or compute loss on aligned overlap.
    print(f"OK: wav_hat length {wav_hat.shape[1]} = {wav_hat.shape[1] // 512} frames × 512 hop")

    # Value range at init (random weights - should be finite, not exploding)
    print(f"wav_hat stats: min={wav_hat.min():.4f} max={wav_hat.max():.4f} mean={wav_hat.mean():.4f}")
