# Supertonic Training Code (KSS Korean Single-Speaker)

Open-source reproduction of the SupertonicTTS 3-stage training pipeline on the KSS dataset (RTX 3090 × 1).

```
training/
├── data/
│   ├── spectrogram.py       SpecProcessor (1253-dim: log-STFT 1025 + log-mel 228)
│   ├── kss.py               KSSDataset (random-crop, AE) + KSSFullUtteranceDataset (TTL/DP) + index builder
│   ├── kss_index.json       12,854 utts × 12.86 h, built by `python -m training.data.kss --build_index`
│   └── ttl_dataset.py       Latent-based dataset for Stage 2/3 (loads cached latents + tokenizes text)
│
├── models/
│   ├── common.py            ConvNeXt1D (causal/symmetric replicate pad)
│   ├── ae_encoder.py        10-layer ConvNeXt encoder, idim=1253 → odim=24   (25.56 M)
│   ├── ae_decoder.py        Pure AE decoder (mirror; matches vocoder.onnx 1.68e-6)  (25.34 M)
│   ├── ae.py                SpecProcessor + Encoder + Decoder wrapper        (50.89 M)
│   ├── discriminators.py    MPD(5) + MRD(3) = 8 heads                        (41.36 M)
│   ├── style_encoder.py     StyleEncoderTTL [B,50,256] + StyleEncoderDP [B,8,16] (2 attn layers, paper-faithful)
│   └── vector_field_cfg.py  VectorFieldCFG wrapper (classifier-free guidance)
│
├── losses/
│   ├── ae_losses.py         multi-res mel L1 + LSGAN + feature matching
│   └── flow_matching.py     CFM loss + UncondMasker + batch_expand + SPFM filter
│
└── scripts/
    ├── train_ae.py                   Stage 1: AE-GAN trainer
    ├── cache_latents.py              (post-AE) dump z_1 per utterance + mean/std stats
    ├── train_ttl.py                  Stage 2: text-to-latent flow matching (uses cached latents)
    ├── train_dp.py                   Stage 3: duration predictor (L1 on total seconds)
    ├── export_onnx.py                export trained ckpts to ONNX (5 files: 4 standard + vector_estimator_cfg)
    ├── extract_voice_style.py        reference wav → voice_styles JSON (using trained encoders)
    ├── infer_cfg.py                  end-to-end TTS with CFG (paper-quality inference in pure PyTorch)
    ├── verify_decoder_weights.py     sanity: load ONNX vocoder into our AEDecoder (1.68e-6)
    └── verify_export_roundtrip.py    sanity: export_onnx.py round-trips vs shipped ONNX (0~7e-7)
```

Inference-side modules (forward = training forward) are imported from `analysis/torch_*.py`
which was verified bit-close to the released ONNX in the reverse-engineering phase. TTL/DP
training scripts `sys.path.insert` those so we train the exact same forward as the released
inference graphs.

## Setup

```bash
conda create -n supertonic python=3.11 -y
conda activate supertonic
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r training/requirements.txt

# build the data index once (scans all KSS wavs, validates SR=44.1k)
python -m training.data.kss --build_index
# → training/data/kss_index.json
```

## Stage-by-stage run

```bash
# === Stage 1: AE (GAN). Heavy. Paper 1.5M step on 4×4090.
#      Single 3090: batch 16, 300k step ≈ 1.5-2 days expected.
python -m training.scripts.train_ae --smoke                 # 500-step sanity (3 min)
python -m training.scripts.train_ae --batch_size 16 --steps 300000 --out_dir training/runs/ae_v1

# === (post-Stage-1) Cache AE latents for all KSS utterances + compute mean/std.
python -m training.scripts.cache_latents \
    --ckpt    training/runs/ae_v1/ckpt_step00300000.pt \
    --out_dir training/runs/ae_v1/cache                       # ~400 MB on disk

# === Stage 2: TTL (flow matching).
#      Paper 700k step, LR halve every 300k. Single 3090: batch 8 × K_e=6 → eff 48.
python -m training.scripts.train_ttl --cache_dir training/runs/ae_v1/cache --smoke
python -m training.scripts.train_ttl --cache_dir training/runs/ae_v1/cache --steps 150000 \
    --out_dir training/runs/ttl_v1

# === Stage 3: DP (tiny). Paper 3k step.
python -m training.scripts.train_dp --cache_dir training/runs/ae_v1/cache --steps 3000 \
    --out_dir training/runs/dp_v1

# === Export all trained models to ONNX (matching assets/onnx/*.onnx format).
python -m training.scripts.export_onnx \
    --ae_ckpt  training/runs/ae_v1/ckpt_step00300000.pt \
    --stats    training/runs/ae_v1/cache/stats.pt \
    --ttl_ckpt training/runs/ttl_v1/ckpt_step00150000.pt \
    --dp_ckpt  training/runs/dp_v1/ckpt_step00003000.pt \
    --out_dir  training/runs/exported/

# Now the exported models plug into existing inference:
python py/example_onnx.py --onnx-dir training/runs/exported/ --lang ko --text "안녕하세요."

# === Extract a voice style from any reference audio (→ voice_styles JSON).
python -m training.scripts.extract_voice_style \
    --wav reference.wav \
    --ae_ckpt training/runs/ae_v1/ckpt_step00300000.pt \
    --ttl_ckpt training/runs/ttl_v1/ckpt_step00150000.pt \
    --dp_ckpt  training/runs/dp_v1/ckpt_step00003000.pt \
    --out voice_styles/my_voice.json --name "MyVoice"

# === Paper-quality inference with CFG (PyTorch, not ONNX).
python -m training.scripts.infer_cfg \
    --ae_ckpt  training/runs/ae_v1/ckpt_step00300000.pt \
    --stats    training/runs/ae_v1/cache/stats.pt \
    --ttl_ckpt training/runs/ttl_v1/ckpt_step00150000.pt \
    --dp_ckpt  training/runs/dp_v1/ckpt_step00003000.pt \
    --voice_style voice_styles/my_voice.json \
    --text "안녕하세요, 제가 훈련한 모델입니다." \
    --lang ko --cfg_scale 3.0 --total_step 5 \
    --out out.wav
```

## Classifier-Free Guidance (CFG)

The paper strongly recommends **CFG with scale=3** for quality. Two ways to use it:

1. **Pure PyTorch** (simplest): use `training/scripts/infer_cfg.py` as above.
2. **ONNX with existing runners**: `export_onnx.py` emits `vector_estimator_cfg.onnx`
   which takes an extra `cfg_scale [B]` input. Set it to 0 for identical behavior to
   the original `vector_estimator.onnx`; set to 3 for paper-quality CFG. Requires a
   small patch to `py/helper.py` to add the new input (see the exported ONNX signature).

During training, `train_ttl.py` applies `UncondMasker` with probs (4%/1%/95%) that
train the model for conditional AND unconditional generation simultaneously.

## Sanity checks (no training required)

```bash
# 1. Module forward matches ONNX (run anytime):
python -m training.scripts.verify_decoder_weights      # vocoder decoder ↔ AEDecoder
python -m training.scripts.verify_export_roundtrip     # full export pipeline via released weights
#   → vocoder/DP/text_encoder: exact match (0 diff), vector_estimator: ~1e-7
```

## Design notes (what's non-obvious)

- **idim=1253** = STFT magnitude bins (1025 for n_fft=2048) + mel bins (228) concatenated along channel.
- **Padding**: vocoder/AE decoder uses `causal` replicate pad; most other ConvNeXt blocks use `symmetric` replicate. See `analysis/HANDOFF.md` for the exhaustive list.
- **AE decoder in training vs inference**: `vocoder.onnx` bakes in 6× un-chunk and z-score de-normalization. We separate these: `AEDecoder` (training) works purely in AE latent space, normalization is applied in `ttl_dataset.prepare_ttl_latent`.
- **latent_mean/std**: computed over the entire training set during `cache_latents.py`. Stored in `cache/stats.pt`. Used to normalize z_1 before TTL flow matching and recovered in vocoder inference.
- **normalizer.scale = 0.25**: applied after z-score standardization to bring latent into a smaller range for flow matching numerical stability.
- **Style encoder** (`StyleEncoderTTL`): reference AE latent → chunk-compress 6× → ConvNeXt 6L → GST-style attention pool with 50 learnable query tokens → [B, 50, 256]. DP variant is the same pattern with smaller dims (8 tokens × 16).
- **CFG via UncondMasker**: per-sample drop with prob 0.04 (both) or 0.01 (text only); replaced with learnable uncond tokens + N(0, 0.1²) noise.
- **Batch expander (K_e=6)**: after text/style encoders run once per sample, we duplicate the conditioning 6× and pair each copy with a different (z_0, t). Per paper this is 6× cheaper than increasing batch size, with similar effect on flow matching convergence.
- **SPFM** (optional, provided in `flow_matching.py` but not wired into `train_ttl.py`): post-warmup binary filter that trains unconditionally on samples where `L_cond > L_uncond`. Mitigates noisy labels.

## Export for inference

Once all three stages converge you can trace each model to ONNX with the exact same input/output
contract as the shipped `assets/onnx/*.onnx`. The forward methods are already verified byte-close,
so the resulting ONNX will plug directly into the existing `py/helper.py` / language-specific
runtimes. (Export script TBD — straightforward once weights are trained.)

## Verification during training

- Save audio samples every N steps in `train_ae.py` (`runs/*/samples/step*.wav`) — listen to hear when reconstruction starts working.
- TensorBoard under `runs/*/tb/` for all loss curves.
- `training/scripts/verify_decoder_weights.py` confirms our `AEDecoder` implementation still matches shipped vocoder.onnx — run anytime for regression check.
