"""Print parameter counts for AE / TTL / DP and compare against paper baseline
(Table 5: #DP 0.5M, #T2F 18.5M, #F2S 25M, #All 44M) and against the shipped
Hugging Face ONNX (~66M with 2x wider TextEncoder/VF).

Run:
    python -m training.scripts.count_params
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from training.models.ae import SpeechAutoencoder
from training.models.style_encoder import (
    StyleEncoderTTL, StyleEncoderTTLPaper,
    StyleEncoderDP, StyleEncoderDPPaper,
)
from torch_text_encoder import TextEncoder, TextEncoderPaper  # type: ignore
from torch_vector_estimator import VectorField                 # type: ignore
from torch_duration_predictor import DurationPredictor         # type: ignore


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def fmt(n):
    return f"{n/1e6:7.3f} M"


def main():
    print("=" * 64)
    print("Paper baseline (this repo's default — `train_ttl.py` no flags)")
    print("=" * 64)
    ae = SpeechAutoencoder(spec_mode="mel")
    te_p = TextEncoderPaper(style_dim=128, attn_type="rope")
    se_ttl_p = StyleEncoderTTLPaper()
    vf_p = VectorField(dim=256, latent_dim=144, n_outer=4, time_dim=64,
                       inter=1024, ksz=5, text_dim=128, style_dim=128,
                       learn_style_prototype=False)
    dp = DurationPredictor(attn_type="rope")
    se_dp_p = StyleEncoderDPPaper()

    ae_enc = n_params(ae.encoder)
    ae_dec = n_params(ae.decoder)
    te = n_params(te_p)
    se_t = n_params(se_ttl_p)
    vf = n_params(vf_p)
    dp_n = n_params(dp)
    se_d = n_params(se_dp_p)

    print(f"  AE encoder (training only)    {fmt(ae_enc)}")
    print(f"  AE decoder (= vocoder)        {fmt(ae_dec)}   paper #F2S = 25 M")
    print(f"  TextEncoder (dim=128)         {fmt(te)}")
    print(f"  StyleEncoderTTLPaper          {fmt(se_t)}")
    print(f"  VectorField (dim=256)         {fmt(vf)}")
    print(f"    TTL subtotal                {fmt(te + se_t + vf)}   paper #T2F = 18.5 M")
    print(f"  DP                            {fmt(dp_n)}")
    print(f"  StyleEncoderDPPaper           {fmt(se_d)}")
    print(f"    DP subtotal                 {fmt(dp_n + se_d)}   paper #DP = 0.5 M")
    inference_paper = ae_dec + te + se_t + vf + dp_n + se_d
    full_paper = ae_enc + inference_paper
    print(f"  ----")
    print(f"  Inference total               {fmt(inference_paper)}   paper #All = 44 M")
    print(f"  Full (incl. AE encoder)       {fmt(full_paper)}")

    print()
    print("=" * 64)
    print("Shipped variant (`train_ttl.py --shipped_dim`)")
    print("=" * 64)
    te_s = TextEncoder()                          # default dim=256
    se_ttl_s = StyleEncoderTTL()
    vf_s = VectorField()                          # default dim=512
    dp_s_n = n_params(DurationPredictor(attn_type="relpos"))
    se_dp_s = n_params(StyleEncoderDP())

    te_sn = n_params(te_s)
    se_t_sn = n_params(se_ttl_s)
    vf_sn = n_params(vf_s)

    print(f"  AE decoder (= vocoder)        {fmt(ae_dec)}")
    print(f"  TextEncoder (dim=256)         {fmt(te_sn)}")
    print(f"  StyleEncoderTTL               {fmt(se_t_sn)}")
    print(f"  VectorField (dim=512)         {fmt(vf_sn)}")
    print(f"    TTL subtotal                {fmt(te_sn + se_t_sn + vf_sn)}")
    print(f"  DP                            {fmt(dp_s_n)}")
    print(f"  StyleEncoderDP                {fmt(se_dp_s)}")
    inference_shipped = ae_dec + te_sn + se_t_sn + vf_sn + dp_s_n + se_dp_s
    print(f"  ----")
    print(f"  Inference total               {fmt(inference_shipped)}   shipped ONNX ≈ 66 M")


if __name__ == "__main__":
    main()
