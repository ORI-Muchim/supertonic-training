# Supertonic Training Pipeline (Reverse-Engineered)

Open-source reproduction of the [SupertonicTTS](https://github.com/supertone-inc/supertonic)
3-stage training pipeline, built from the released ONNX models and the paper.

Two layers of content:

1. **`analysis/`** — PyTorch reimplementations of the 4 shipped ONNX modules
   (`duration_predictor`, `text_encoder`, `vector_estimator`, `vocoder`), verified
   **bit-close** against the official ONNX (max \|Δ\| ≤ 3e-6 on final waveform).
   These are the *inference* modules and define the exact forward graph used
   during training.

2. **`training/`** — Full 3-stage training code (AE-GAN → TTL flow-matching → DP)
   on the KSS Korean single-speaker dataset, plus `export_onnx.py` that rebuilds
   the shipped 5-file ONNX bundle. Round-trip verification confirms the
   exported ONNX matches the official release exactly for 3 modules and to
   FP precision (6.6e-7) for `vector_estimator`.

## What this isn't

- **Not** a fork of supertone-inc/supertonic. It's a companion repo with
  training code; inference stays in the upstream repo.
- **Not** distributed with Supertone's ONNX weights. Download them separately
  (see Setup).

## Setup

```bash
conda create -n supertonic python=3.11 -y
conda activate supertonic
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download the official release assets (ONNX + voice_styles + tts.json +
unicode_indexer.json) from
[Supertone/supertonic-2](https://huggingface.co/Supertone/supertonic-2)
and place them at repo root:

```
assets/
├── onnx/
│   ├── duration_predictor.onnx
│   ├── text_encoder.onnx
│   ├── vector_estimator.onnx
│   ├── vocoder.onnx
│   ├── tts.json
│   └── unicode_indexer.json
└── voice_styles/
    ├── F1.json … F5.json
    └── M1.json … M5.json
```

## Verify the reverse engineering (no training required)

```bash
# 1) End-to-end: PyTorch vs ONNX on 4 languages, all stages
python analysis/verify_end_to_end.py
# → duration  max|Δ|≈1e-6
#   text_emb  max|Δ|≈1e-5
#   latent    max|Δ|≈1e-4
#   wav       max|Δ|≈3e-3  (mean 1e-6 — FP precision)

# 2) AE decoder weights ↔ shipped vocoder.onnx
python -m training.scripts.verify_decoder_weights
# → max|Δ|=1.68e-6

# 3) Full export pipeline round-trip
python -m training.scripts.verify_export_roundtrip
# → vocoder / dp / text_encoder: 0.00e+00 (bit-exact)
#   vector_estimator:             6.56e-7
```

## Train from scratch (KSS)

See `training/README.md` for the 3-stage recipe. Rough compute on RTX 3090:

| Stage | Paper | 3090 single-GPU |
|---|---|---|
| AE-GAN | 1.5 M steps × 4×4090 | batch 16, 300 k steps ≈ 1.5–2 days |
| TTL (flow matching) | 700 k steps | batch 8, K_e=6 (eff 48), 150 k ≈ 1 day |
| DP | 3 k steps | < 10 min |

## Credits

Built from the public release of
[SupertonicTTS](https://github.com/supertone-inc/supertonic) (MIT) and the
[accompanying paper](https://huggingface.co/Supertone/supertonic-2). Upstream
MIT license preserved in `LICENSE`; original copyright belongs to Supertone
Inc. — this repo's additions are reverse-engineering notes and a KSS training
pipeline that together reproduce the forward graph used by the released
models.
