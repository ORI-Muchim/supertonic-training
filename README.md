# Supertonic Training Pipeline (Reverse-Engineered)

Educational reverse engineering of [SupertonicTTS](https://github.com/supertone-inc/supertonic)
(arXiv [2503.23108](https://arxiv.org/abs/2503.23108)), built from the released ONNX models
and the paper.

**This is not an official Supertone repo. No shipped weights are redistributed.**
All credit for the model goes to the Supertone team.

---

> ### ⚠️ Heads-up before you clone this for "training a TTS"
>
> **Running the training code here from scratch will NOT produce a usable TTS
> model.** The paper's recipe assumes ~100 GPU-days on 4×4090 plus an
> undisclosed multi-lingual multi-speaker dataset, and the practical details
> (LR schedule, discriminator warmup, data augmentation, loss-weight schedule,
> text normalizer, etc.) are not fully specified in the paper.
>
> At single-GPU scale, GAN dynamics collapse (discriminator wins, gradient
> explodes) and the fresh AE encoder's output distribution doesn't line up
> with the shipped vocoder, so end-to-end synthesis from a locally-trained
> stack sounds garbled.
>
> **This repo is for:** reverse-engineering study of the Supertonic-2
> architecture, bit-close forward-graph verification, and a reference
> implementation of the 3-stage training loss/graph wiring.
>
> **For actual voice cloning**, use the companion repo
> [supertonic.embed](https://github.com/kdrkdrkdr/supertonic.embed) — it
> optimizes `style_ttl`/`style_dp` directly against a reference clip via
> HuBERT perceptual loss on top of the frozen shipped model, and produces
> same-voice-level quality in ~15 minutes on a 3090 with zero training.

---

Two layers of content:

1. **`analysis/`** — PyTorch reimplementations of the 4 shipped ONNX modules
   (`duration_predictor`, `text_encoder`, `vector_estimator`, `vocoder`), verified
   **bit-close** against the official ONNX (max \|Δ\| ≤ 3e-6 on final waveform).
   These are the *inference* modules and define the exact forward graph used
   during training.

2. **`training/`** — Full 3-stage training code (AE-GAN → TTL flow-matching → DP)
   runnable on the KSS Korean single-speaker dataset, plus `export_onnx.py` that
   rebuilds the shipped 5-file ONNX bundle. Round-trip verification confirms the
   exported ONNX matches the official release exactly for 3 modules and to
   FP precision (6.6e-7) for `vector_estimator`.

## What works and what doesn't

**Works well (well-tested):**
- Forward graph is bit-close to shipped ONNX (all 4 modules).
- `verify_end_to_end.py`, `verify_decoder_weights.py`, `verify_export_roundtrip.py`
  reproduce quality metrics on any machine with the shipped weights.
- Fine-tune mode (Stage-2, Stage-3): load shipped `text_encoder.onnx` +
  `vector_estimator.onnx` + `duration_predictor.onnx` as frozen init, train only
  the style encoders on ~hours of compute.

**Works with caveats:**
- **From-scratch training** (train_ae.py GAN-mode) runs and reduces loss on KSS,
  but hitting paper quality requires paper-scale compute (~100 GPU-days for
  Stage 1 at 4×4090) and data that isn't publicly disclosed. This pipeline
  demonstrates the forward graph and training loss wiring, not a plug-and-play
  single-GPU paper-quality reproduction.
- `train_ae_ft.py` (Stage-1 fine-tune with frozen shipped vocoder decoder) is the
  pragmatic single-GPU path — it reconstructs KSS via the shipped decoder as the
  target. Mel loss descends to ~1.0 in 60k steps (~40 min on 3090). Still
  noticeably below shipped encoder quality because we lack the shipped encoder
  weights to warm-start from.

**Doesn't work out of the box:**
- **Single-speaker voice cloning via this repo's training code alone.** The
  distribution gap between our freshly trained AE encoder and the shipped
  vocoder's expected latent distribution means end-to-end synthesis from an
  encoder trained here produces muddy output. For voice cloning with the shipped
  model, the practical approach is direct style-vector optimization against a
  reference — see the companion
  [supertonic.embed](https://github.com/kdrkdrkdr/supertonic.embed) repo, which
  optimizes `style_ttl`/`style_dp` via HuBERT perceptual loss using only shipped
  weights and produces high-quality single-speaker voices in ~15 minutes on a 3090.

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

## Training recipes

### Paper-faithful from-scratch (reference only — paper-scale compute required)

See `training/README.md` for the 3-stage recipe. Compute estimate on a single
RTX 3090 (for reference — this is not the practical path):

| Stage | Paper | 3090 estimated |
|---|---|---|
| AE-GAN | 1.5 M steps × 4×4090 | batch 16, 300 k steps ≈ 1.5–2 days |
| TTL (flow matching) | 700 k steps | batch 8, K_e=6 (eff 48), 150 k ≈ 1 day |
| DP | 3 k steps | < 10 min |

### Fine-tune from shipped weights (practical path)

```bash
# Stage 1: train AEEncoder with frozen shipped vocoder decoder (~40 min at batch 16)
python -m training.scripts.train_ae_ft --steps 60000 --out_dir training/runs/ae_ft

# Cache AE latents (~1 min)
python -m training.scripts.cache_latents \
    --ckpt training/runs/ae_ft/ckpt_step00060000.pt \
    --out_dir training/runs/ae_ft/cache --fp16

# Stage 2: StyleEncoderTTL only, shipped TE+VF frozen (~40 min)
python -m training.scripts.train_ttl --cache_dir training/runs/ae_ft/cache \
    --steps 25000 --out_dir training/runs/ttl_ft

# Stage 3: StyleEncoderDP only, shipped DP frozen (~1 min)
python -m training.scripts.train_dp --cache_dir training/runs/ae_ft/cache \
    --steps 5000 --out_dir training/runs/dp_ft

# Extract a voice style JSON from any reference wav
python -m training.scripts.extract_voice_style \
    --wav <reference.wav> \
    --ae_ckpt  training/runs/ae_ft/ckpt_step00060000.pt \
    --stats    training/runs/ae_ft/cache/stats.pt \
    --ttl_ckpt training/runs/ttl_ft/ckpt_step00025000.pt \
    --dp_ckpt  training/runs/dp_ft/ckpt_step00005000.pt \
    --out assets/voice_styles/MyVoice.json --name MyVoice
```

Fine-tune uses `load_*_weights` (from `analysis/torch_*.py`) to load the shipped
ONNX weights directly into the PyTorch modules. This gives you paper-quality
text/flow/vocoder for free; only the style encoders (≈1.5M + 0.15M params) are
trained. Total cost: < 2 hours on a 3090 for KSS single-speaker.

Practical caveat: the from-scratch AEEncoder still doesn't match the shipped
encoder's output distribution well enough for clean end-to-end synthesis from
the fine-tuned stack alone. For actual high-quality voice cloning, use
`supertonic.embed` (link above), which sidesteps encoder training entirely.

## License & credits

Model architecture, weights, and the original paper are property of
Supertone Inc. (MIT-licensed code, OpenRAIL-M licensed weights).

- Paper: arXiv [2503.23108](https://arxiv.org/abs/2503.23108)
- Official code: [supertone-inc/supertonic](https://github.com/supertone-inc/supertonic)
- Weights: [Supertone/supertonic-2](https://huggingface.co/Supertone/supertonic-2)
- Related supporting papers:
  arXiv [2509.11084](https://arxiv.org/abs/2509.11084) (LARoPE),
  arXiv [2509.19091](https://arxiv.org/abs/2509.19091) (SPFM)

The code in this repo is distributed under MIT (see `LICENSE`, preserved from
upstream). The additions here are reverse-engineering notes and training
scripts, intended for research and educational use.

`py/helper.py` is a bundled copy of
[supertone-inc/supertonic/py/helper.py](https://github.com/supertone-inc/supertonic)
(MIT) — kept here unchanged so the verification scripts run without cloning
the full upstream repo.
