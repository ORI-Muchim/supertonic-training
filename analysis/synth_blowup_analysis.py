"""Analyze why cfg=3 synth outputs sound 'blown out' vs cfg=1.

For each v2 wav: peak/clipping stats + mel spectrogram + waveform plot.
"""
from __future__ import annotations
import sys, numpy as np, soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
import librosa

ROOT = Path(__file__).resolve().parents[1]

SR = 44100
N_FFT = 2048
HOP = 512
N_MELS = 80


def to_mel(w):
    if w.ndim > 1: w = w.mean(axis=-1)
    s = librosa.feature.melspectrogram(
        y=w.astype(np.float32), sr=SR, n_fft=N_FFT, hop_length=HOP,
        win_length=N_FFT, n_mels=N_MELS, fmin=0, fmax=SR // 2,
    )
    return np.log(np.clip(s, 1e-5, None))


def main():
    files = [
        ("v2_b_cfg1s16  (natural)",  "out_e2e_v2_b_cfg1s16.wav"),
        ("v2_b_cfg3s5   (blown)",    "out_e2e_v2_b_cfg3s5.wav"),
        ("v2_a_cfg3s5   (worst)",    "out_e2e_v2_a_cfg3s5.wav"),
    ]

    # Reference: real KSS (refsame text)
    real_wav, _ = sf.read(str(ROOT / "archive/kss/1/1_0000.wav"), dtype="float32")
    mel_real = to_mel(real_wav)

    print("=== amplitude / clipping stats ===")
    print(f"{'label':30s} {'peak':>8s} {'rms':>8s} {'>0.9':>8s} {'>0.99':>8s} {'sat%':>8s}")
    print(f"{'REAL KSS 1_0000.wav':30s} {np.abs(real_wav).max():>8.3f} {np.sqrt((real_wav**2).mean()):>8.3f} "
          f"{(np.abs(real_wav)>0.9).mean()*100:>7.2f}% {(np.abs(real_wav)>0.99).mean()*100:>7.2f}% "
          f"{(np.abs(real_wav)>=1.0-1e-4).mean()*100:>7.2f}%")
    for label, fn in files:
        w, _ = sf.read(str(ROOT / fn), dtype="float32")
        peak = float(np.abs(w).max())
        rms = float(np.sqrt((w**2).mean()))
        n9  = (np.abs(w) > 0.9).mean() * 100
        n99 = (np.abs(w) > 0.99).mean() * 100
        sat = (np.abs(w) >= 1.0 - 1e-4).mean() * 100
        print(f"{label:30s} {peak:>8.3f} {rms:>8.3f} {n9:>7.2f}% {n99:>7.2f}% {sat:>7.2f}%")

    # === plot ===
    fig, axs = plt.subplots(4, 2, figsize=(14, 12))

    # Row 0: REAL
    axs[0, 0].plot(real_wav, lw=0.3)
    axs[0, 0].set_ylim(-1.05, 1.05); axs[0, 0].set_title("REAL KSS — waveform")
    axs[0, 0].axhline(1.0, color="r", lw=0.5, ls="--"); axs[0, 0].axhline(-1.0, color="r", lw=0.5, ls="--")
    im = axs[0, 1].imshow(mel_real, aspect="auto", origin="lower", cmap="magma", vmin=-11.5, vmax=8)
    axs[0, 1].set_title("REAL KSS — log-mel")

    for i, (label, fn) in enumerate(files):
        w, _ = sf.read(str(ROOT / fn), dtype="float32")
        m = to_mel(w)
        axs[i + 1, 0].plot(w, lw=0.3)
        axs[i + 1, 0].set_ylim(-1.05, 1.05); axs[i + 1, 0].set_title(f"{label} — waveform")
        axs[i + 1, 0].axhline(1.0, color="r", lw=0.5, ls="--"); axs[i + 1, 0].axhline(-1.0, color="r", lw=0.5, ls="--")
        axs[i + 1, 1].imshow(m, aspect="auto", origin="lower", cmap="magma", vmin=-11.5, vmax=8)
        axs[i + 1, 1].set_title(f"{label} — log-mel")

    for ax in axs.ravel():
        ax.set_xticks([])

    plt.tight_layout()
    out = ROOT / "blowup_analysis.png"
    plt.savefig(str(out), dpi=120, bbox_inches="tight")
    print(f"\nplot saved -> {out}")


if __name__ == "__main__":
    main()
