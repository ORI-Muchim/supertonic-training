"""Compare mel spectrograms: real KSS vs AE-only vs E2E (TTL+DP).

Loads:
  - Real wav: archive/kss/1/1_0000.wav  (idx 0, text "그는 괜찮은 척하려고...")
  - AE-only: cached z_ae [24, T] → AEDecoder → wav
  - E2E:     out_e2e_b_refsame.wav  (TTL with same text)

Computes 80-mel log-mel for each, plots side-by-side, prints stats.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from training.models.ae_decoder import AEDecoder

SR = 44100
N_FFT = 2048
HOP = 512
N_MELS = 80


def to_mel(wav: np.ndarray) -> np.ndarray:
    """Returns log-mel [N_MELS, T_frames]."""
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    s = librosa.feature.melspectrogram(
        y=wav.astype(np.float32), sr=SR, n_fft=N_FFT, hop_length=HOP,
        win_length=N_FFT, n_mels=N_MELS, fmin=0, fmax=SR // 2,
    )
    return np.log(np.clip(s, 1e-5, None))


def stats(mel: np.ndarray, label: str) -> dict:
    # Spectral flatness on the linear (pre-log) magnitude — but use mel as proxy: ratio of geo mean to arith mean
    mel_lin = np.exp(mel)
    geo = np.exp(np.mean(np.log(mel_lin + 1e-12), axis=0))
    arith = np.mean(mel_lin, axis=0)
    flatness = np.mean(geo / (arith + 1e-12))
    # High-freq energy ratio: top 25% mel bands / total
    hf = mel_lin[3 * N_MELS // 4:].sum() / mel_lin.sum()
    # Dynamic range
    dr = mel.max() - mel.min()
    return {
        "label": label,
        "shape": mel.shape,
        "min": float(mel.min()), "max": float(mel.max()), "mean": float(mel.mean()),
        "dynamic_range_db": 10 * np.log10(np.exp(dr)),  # approximate dB
        "spectral_flatness": float(flatness),
        "high_freq_energy_frac": float(hf),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- 1) real KSS wav (idx 0) --
    real_path = ROOT / "archive" / "kss" / "1" / "1_0000.wav"
    real_wav, sr_r = sf.read(str(real_path), dtype="float32")
    assert sr_r == SR, f"expected SR={SR}, got {sr_r}"

    # -- 2) AE-only: cache → AEDecoder --
    cache = ROOT / "training" / "runs" / "ae_paper_audit_crop1s_mean" / "cache"
    z_ae = torch.load(cache / "latents" / "0000000.pt", weights_only=False).float().to(device)  # [24, T]
    ae_ck = torch.load(ROOT / "training" / "runs" / "ae_paper_audit_crop1s_mean" / "ckpt_step01500000.pt",
                       map_location=device, weights_only=False)
    dec = AEDecoder().to(device)
    dec_sd = {k.replace("decoder.", "", 1): v for k, v in ae_ck["ae"].items() if k.startswith("decoder.")}
    dec.load_state_dict(dec_sd, strict=False); dec.eval()
    with torch.no_grad():
        ae_wav = dec(z_ae.unsqueeze(0)).squeeze(0).cpu().numpy().astype(np.float32)
    # match real length
    n = min(len(real_wav), len(ae_wav))
    ae_wav = ae_wav[:n]
    real_wav_match = real_wav[:n]

    # -- 3) E2E synth output (TTL+DP) --
    e2e_wav, sr_e = sf.read(str(ROOT / "out_e2e_b_refsame.wav"), dtype="float32")

    # -- mels --
    mel_real = to_mel(real_wav_match)
    mel_ae   = to_mel(ae_wav)
    mel_e2e  = to_mel(e2e_wav)

    print("=== STATS ===")
    for m, lbl in [(mel_real, "REAL"), (mel_ae, "AE-only"), (mel_e2e, "E2E (TTL+DP)")]:
        s = stats(m, lbl)
        print(f"{lbl:14s} shape={s['shape']}  min={s['min']:6.2f} max={s['max']:6.2f} "
              f"mean={s['mean']:6.2f} DR≈{s['dynamic_range_db']:5.1f}dB "
              f"flatness={s['spectral_flatness']:.4f} HF_frac={s['high_freq_energy_frac']:.3f}")

    # -- mel L1 distance vs real (truncate to min frames) --
    T = min(mel_real.shape[1], mel_ae.shape[1], mel_e2e.shape[1])
    mae_ae  = float(np.mean(np.abs(mel_real[:, :T] - mel_ae[:, :T])))
    mae_e2e = float(np.mean(np.abs(mel_real[:, :T] - mel_e2e[:, :T])))
    print(f"\nmel-MAE  REAL vs AE-only: {mae_ae:.4f}")
    print(f"mel-MAE  REAL vs E2E    : {mae_e2e:.4f}")

    # -- plot --
    fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
    vmin = min(mel_real.min(), mel_ae.min(), mel_e2e.min())
    vmax = max(mel_real.max(), mel_ae.max(), mel_e2e.max())
    for ax, mel, title in zip(axs, [mel_real, mel_ae, mel_e2e],
                              ["REAL KSS (1_0000.wav)",
                               "AE-only (z_ae → decoder)",
                               "E2E paper-faithful (TTL 700k + DP 3k + AE)"]):
        im = ax.imshow(mel, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="magma")
        ax.set_title(title); ax.set_ylabel("mel band")
    axs[-1].set_xlabel("frame")
    plt.colorbar(im, ax=axs.ravel().tolist(), label="log-mel")
    out = ROOT / "mel_compare.png"
    plt.savefig(str(out), dpi=120, bbox_inches="tight")
    print(f"\nplot saved -> {out}")


if __name__ == "__main__":
    main()
