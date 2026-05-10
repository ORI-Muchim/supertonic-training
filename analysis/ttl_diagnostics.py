"""TTL diagnostics — separate inference-pipeline bugs from underfit.

Tests:
  1. GT latent decode: take ground-truth z_ttl (same transform TTL was trained to predict),
     invert it, decode through AE — compare mel vs AE-only baseline.
     If matches AE-only: inversion/decode pipeline is OK.
  2. TTL output distribution: encode reference, run TTL ODE, compare resulting z_ttl
     channel-wise mean/std against GT z_ttl from same utterance.
     Large mismatch -> normalization/training distribution issue.
  3. CFG/steps sweep already done implicitly (a/c/d files exist).

Uses ref_idx=0 (text "그는 괜찮은 척하려고...") so we can compare to the GT.
"""
from __future__ import annotations
import sys, json
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import librosa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "py"))

from training.models.ae_decoder import AEDecoder
from training.models.style_encoder import StyleEncoderTTLPaper
from training.data.ttl_dataset import (
    prepare_ttl_latent, invert_ttl_latent, TTL_NORMALIZER_SCALE,
)
from helper import UnicodeProcessor                          # type: ignore
from torch_text_encoder import TextEncoderPaper              # type: ignore
from torch_vector_estimator import VectorField               # type: ignore

SR = 44100
HOP = 512
KC = 6
N_FFT = 2048
N_MELS = 80


def to_mel(wav):
    if wav.ndim > 1: wav = wav.mean(axis=-1)
    s = librosa.feature.melspectrogram(
        y=wav.astype(np.float32), sr=SR, n_fft=N_FFT, hop_length=HOP,
        win_length=N_FFT, n_mels=N_MELS, fmin=0, fmax=SR // 2,
    )
    return np.log(np.clip(s, 1e-5, None))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = ROOT / "training" / "runs" / "ae_paper_audit_crop1s_mean" / "cache"
    ae_ckpt = ROOT / "training" / "runs" / "ae_paper_audit_crop1s_mean" / "ckpt_step01500000.pt"
    ttl_ckpt = ROOT / "training" / "runs" / "ttl_paper_700k_paperfix2" / "ckpt_step00700000.pt"

    # -- decoder + stats --
    ae_ck = torch.load(str(ae_ckpt), map_location=device, weights_only=False)
    dec = AEDecoder().to(device)
    dec_sd = {k.replace("decoder.", "", 1): v for k, v in ae_ck["ae"].items() if k.startswith("decoder.")}
    dec.load_state_dict(dec_sd, strict=False); dec.eval()
    stats = torch.load(cache / "stats.pt", weights_only=False)
    mean = stats["mean"].float().to(device); std = stats["std"].float().to(device)

    # -- ground-truth latent for idx 0 --
    z_ae = torch.load(str(cache / "latents" / "0000000.pt"), weights_only=False).float().to(device)  # [24, T]

    # -- TTL paper-faithful models --
    ck = torch.load(str(ttl_ckpt), map_location=device, weights_only=False)
    te = TextEncoderPaper(style_dim=128).to(device); te.load_state_dict(ck["text_encoder"]); te.eval()
    se = StyleEncoderTTLPaper().to(device);          se.load_state_dict(ck["style_encoder"]);   se.eval()
    vf = VectorField(dim=256, latent_dim=144, n_outer=4, time_dim=64,
                     inter=1024, ksz=5, text_dim=128, style_dim=128,
                     learn_style_prototype=False).to(device)
    vf.load_state_dict(ck["vector_field"]); vf.eval()
    um = ck["uncond_masker"]
    uncond_text  = um["uncond_text"].to(device)    # [1, 128, 1]
    uncond_style = um["uncond_style"].to(device)   # [1, 50, 128]

    # ===== TEST 1: GT z_ttl roundtrip =====
    print("=" * 60)
    print("TEST 1: GT z_ttl -> invert -> decode  (sanity for inversion pipeline)")
    print("=" * 60)

    # AE-only baseline
    with torch.no_grad():
        wav_ae_only = dec(z_ae.unsqueeze(0)).squeeze(0).cpu().numpy().astype(np.float32)

    # GT z_ttl (same transform TTL was trained to predict)
    z_ttl_gt = prepare_ttl_latent(z_ae, mean, std, scale=TTL_NORMALIZER_SCALE, kc=KC)  # [144, L_ttl]
    # Invert and decode
    z_ae_recovered = invert_ttl_latent(z_ttl_gt, mean, std, scale=TTL_NORMALIZER_SCALE, kc=KC)  # [24, T_recovered]
    print(f"GT z_ae shape:        {tuple(z_ae.shape)}")
    print(f"After prepare→invert: {tuple(z_ae_recovered.shape)}")
    diff = (z_ae - z_ae_recovered[:, :z_ae.shape[-1]]).abs()
    print(f"|z_ae - recovered|  max={diff.max().item():.6f}  mean={diff.mean().item():.6f}")
    with torch.no_grad():
        wav_gt_roundtrip = dec(z_ae_recovered[:, :z_ae.shape[-1]].unsqueeze(0)).squeeze(0).cpu().numpy().astype(np.float32)

    # Match length
    n = min(len(wav_ae_only), len(wav_gt_roundtrip))
    mel_ae   = to_mel(wav_ae_only[:n])
    mel_rt   = to_mel(wav_gt_roundtrip[:n])
    mae_rt = float(np.mean(np.abs(mel_ae - mel_rt)))
    print(f"mel-MAE   AE-only vs GT-roundtrip = {mae_rt:.4f}")
    print(f"          (should be ~0; nonzero means inversion bug)")

    # ===== TEST 2: TTL output distribution vs GT =====
    print()
    print("=" * 60)
    print("TEST 2: TTL output distribution vs GT z_ttl")
    print("=" * 60)

    text_processor = UnicodeProcessor(str(ROOT / "assets" / "onnx" / "unicode_indexer.json"))
    text_ids_np, text_mask_np = text_processor(["그는 괜찮은 척하려고 애쓰는 것 같았다."], ["ko"])
    text_ids  = torch.from_numpy(text_ids_np).to(device)
    text_mask = torch.from_numpy(text_mask_np.astype("float32")).to(device)

    # ref style from full utterance (z-scored)
    ref = (z_ae - mean.view(-1, 1)) / std.view(-1, 1)
    ref = ref.unsqueeze(0)
    ref_mask = torch.ones(1, 1, ref.shape[-1], device=device)

    # Run TTL Euler ODE for each (cfg, steps) combo
    style_ttl = se(ref, ref_mask)
    text_emb  = te(text_ids, style_ttl, text_mask)
    style_proto = te.reference_key
    text_u  = uncond_text.expand(1, -1, text_emb.shape[-1])
    style_u = uncond_style

    # Match GT length so distributions are comparable
    L_ttl = z_ttl_gt.shape[-1]
    print(f"z_ttl_gt stats (per-channel): mean={z_ttl_gt.mean(-1).mean().item():.3f} "
          f"std={z_ttl_gt.std(-1).mean().item():.3f}  (expect ~0 / ~1 from z-score)")

    for cfg_scale, steps in [(0.0, 5), (1.0, 5), (3.0, 5), (3.0, 16), (5.0, 16)]:
        torch.manual_seed(0)
        z_t = torch.randn(1, 144, L_ttl, device=device)
        lat_mask = torch.ones(1, 1, L_ttl, device=device)
        with torch.no_grad():
            for step in range(steps):
                t_norm = torch.tensor([step / steps], device=device)
                v_cond = vf.velocity(z_t, text_emb, style_ttl, lat_mask, text_mask, t_norm,
                                     style_prototype=style_proto)
                if cfg_scale != 0.0:
                    v_unc = vf.velocity(z_t, text_u, style_u, lat_mask, text_mask, t_norm,
                                        style_prototype=style_proto)
                    v = (1.0 + cfg_scale) * v_cond - cfg_scale * v_unc
                else:
                    v = v_cond
                dt = 1.0 / steps
                z_t = (z_t + dt * v) * lat_mask

        z_pred = z_t.squeeze(0)            # [144, L_ttl]
        # per-channel stats
        ch_mean = z_pred.mean(-1)          # [144]
        ch_std  = z_pred.std(-1)           # [144]
        gt_ch_mean = z_ttl_gt.mean(-1)
        gt_ch_std  = z_ttl_gt.std(-1)
        # also compare to GT
        diff = (z_pred - z_ttl_gt).abs().mean().item()
        # decode + mel
        z_ae_pred = invert_ttl_latent(z_pred, mean, std, scale=TTL_NORMALIZER_SCALE, kc=KC)
        with torch.no_grad():
            wav_pred = dec(z_ae_pred.unsqueeze(0)).squeeze(0).cpu().numpy().astype(np.float32)
        nlen = min(len(wav_ae_only), len(wav_pred))
        mel_pred = to_mel(wav_pred[:nlen])
        mel_mae = float(np.mean(np.abs(mel_ae[:, :mel_pred.shape[-1]] - mel_pred[:, :mel_ae.shape[-1]])))

        print(f"cfg={cfg_scale:.1f} steps={steps:2d}  "
              f"ch_mean={ch_mean.mean().item():+.3f}({gt_ch_mean.mean().item():+.3f})  "
              f"ch_std={ch_std.mean().item():.3f}({gt_ch_std.mean().item():.3f})  "
              f"|z_pred-z_gt|={diff:.3f}  "
              f"mel-MAE_vs_AE={mel_mae:.3f}")

    print()
    print("Interpretation:")
    print("  - If TEST1 mel-MAE ~0   : inversion code OK")
    print("  - If TTL ch_std << GT (~1): velocity estimator output is shrunk → underfit / loss-mask scaling issue")
    print("  - If ch_mean drift large : training-time normalization mismatch")
    print("  - If mel-MAE_vs_AE flat across cfg/steps : underfit (more compute won't help much per knob)")
    print("  - If improves with more steps : ODE truncation; cheap to fix at inference")


if __name__ == "__main__":
    main()
