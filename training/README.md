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
│   ├── ae_encoder.py        10-layer ConvNeXt encoder, idim=228 mel → odim=24
│   ├── ae_decoder.py        Pure AE decoder (mirror; matches vocoder.onnx 1.68e-6)
│   ├── ae.py                SpecProcessor + Encoder + Decoder wrapper
│   ├── discriminators.py    MPD(5 periods 2,3,5,7,11) + MRD(3 FFT 512/1024/2048)
│   ├── style_encoder.py     StyleEncoder{TTL,DP} (shipped) + StyleEncoder{TTL,DP}Paper (out_scale=1.0) + StyleEncoderDPTextPaper (experimental)
│   └── vector_field_cfg.py  VectorFieldCFG wrapper (classifier-free guidance)
│
├── losses/
│   ├── ae_losses.py         45·L_recon + L_adv + 0.1·L_fm  (paper Sec 4.2)
│   └── flow_matching.py     CFM L1 loss + joint UncondMasker (p_uncond=0.05) + batch_expand (K_e=4) + SPFM
│
└── scripts/
    ├── train_ae.py                   Stage 1: AE-GAN trainer (paper recipe, 1.5M iter)
    ├── cache_latents.py              (post-AE) dump z_1 per utterance + mean/std stats
    ├── train_ttl.py                  Stage 2: text-to-latent flow matching (paper-faithful default: RoPE TextEncoder, K_e=4, grad_accum)
    ├── train_dp.py                   Stage 3: duration predictor (paper-faithful: 5-95% ref crop, RoPE SentenceEncoder)
    ├── export_onnx.py                export trained ckpts to ONNX (5 files: 4 standard + vector_estimator_cfg)
    ├── extract_voice_style.py        reference wav → voice_styles JSON (using trained encoders)
    ├── infer_cfg.py                  end-to-end TTS with CFG (shipped 256-dim path)
    ├── synth_ttl_paper.py            paper-faithful E2E synth (128-dim TTL + optional DP)
    ├── synth_ae.py                   AE-only synth (encoder→decoder roundtrip for ceiling diagnosis)
    ├── eval_ae_recon.py              deterministic AE reconstruction eval
    ├── verify_decoder_weights.py     sanity: load ONNX vocoder into our AEDecoder (1.68e-6)
    └── verify_export_roundtrip.py    sanity: export_onnx.py round-trips vs shipped ONNX (0~7e-7)
```

Diagnostics live in `analysis/`:
- `mel_compare.py` — real vs AE-only vs E2E mel-MAE comparison (isolates AE ceiling vs TTL contribution)
- `ttl_diagnostics.py` — GT z_ttl roundtrip + TTL output distribution check (per-channel mean/std vs GT, CFG/steps sweep)

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
# === Stage 1: AE (GAN). Paper 1.5M step on 4×4090, batch 128.
#      Single 3090: batch 16, 1.5M step ≈ ~3 days.
python -m training.scripts.train_ae --smoke                 # 500-step sanity (3 min)
python -m training.scripts.train_ae --batch_size 16 --steps 1500000 --out_dir training/runs/ae_v1

# === (post-Stage-1) Cache AE latents for all KSS utterances + compute mean/std.
python -m training.scripts.cache_latents \
    --ckpt    training/runs/ae_v1/ckpt_step01500000.pt \
    --out_dir training/runs/ae_v1/cache                       # ~400 MB on disk

# === Stage 2: TTL (flow matching, paper-faithful default).
#      Paper 700k step @ batch 64, K_e=4, lr 5e-4 halve@300k.
#      Single 3090: batch 32 × grad_accum 2 → effective batch 64 (paper match).
#      Defaults: RoPE TextEncoder, σ_min=1e-8, p_uncond=0.05 joint, ref crop 0.2-9s ≤½, ref loss mask m.
python -m training.scripts.train_ttl --cache_dir training/runs/ae_v1/cache --smoke
python -m training.scripts.train_ttl \
    --cache_dir training/runs/ae_v1/cache \
    --batch_size 32 --grad_accum 2 --attn_type rope \
    --out_dir training/runs/ttl_v1
# ETA: ~3 days on 3090 (700k optimizer updates @ ~2.5 upd/s).

# === Stage 3: DP (tiny). Paper 3k step @ batch 128.
#      Defaults: paper-faithful StyleEncoderDPPaper + RoPE SentenceEncoder + 5-95% ref crop.
python -m training.scripts.train_dp \
    --cache_dir training/runs/ae_v1/cache \
    --out_dir training/runs/dp_v1
# ETA: ~5 min on 3090.

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

The paper recommends **CFG with scale=3** for quality (paper Sec 3.2.4 / B.2). Three ways to use it:

1. **Paper-faithful PyTorch** (128-dim TTL + DP):
   `training/scripts/synth_ttl_paper.py` — `--cfg_scale` arg, runs Euler ODE with
   joint cond/uncond velocity. Use `--dp_ckpt` to enable DP-predicted duration.
2. **Shipped 256-dim PyTorch**: `training/scripts/infer_cfg.py` (older path,
   loads shipped ONNX dims).
3. **ONNX with existing runners**: `export_onnx.py` emits `vector_estimator_cfg.onnx`
   with extra `cfg_scale [B]` input. Set 0 for vanilla, 3 for CFG.

During training, `train_ttl.py` applies `UncondMasker` with **single joint Bernoulli
p_uncond=0.05** (paper B.2) — both `text_emb` AND `style_ttl` replaced together
with learned uncond tokens. (Older `4%/1%` split was pre-paper-faithful.)

## Sanity checks (no training required)

```bash
# 1. Module forward matches ONNX (run anytime):
python -m training.scripts.verify_decoder_weights      # vocoder decoder ↔ AEDecoder
python -m training.scripts.verify_export_roundtrip     # full export pipeline via released weights
#   → vocoder/DP/text_encoder: exact match (0 diff), vector_estimator: ~1e-7
```

## Design notes (what's non-obvious)

- **AE input = 228-dim mel** (paper Sec 4.2: "228 mel bands", FFT 2048, hop 512). Earlier `idim=1253` (mel+STFT concat) was a non-paper experiment and was reverted.
- **Padding**: vocoder/AE decoder uses `causal` replicate pad; most other ConvNeXt blocks use `symmetric` replicate. See `analysis/HANDOFF.md` for the exhaustive list.
- **AE decoder in training vs inference**: `vocoder.onnx` bakes in 6× un-chunk and z-score de-normalization. We separate these: `AEDecoder` (training) works purely in AE latent space, normalization is applied in `ttl_dataset.prepare_ttl_latent` / `invert_ttl_latent`.
- **latent_mean/std**: computed over the entire training set during `cache_latents.py`. Stored in `cache/stats.pt`. Used to normalize z_1 before TTL flow matching and recovered in vocoder inference.
- **TTL_NORMALIZER_SCALE = 1.0** (paper-faithful): single channel-wise z-score, no extra multiplicative scale. Older 0.25 was a shipped-distribution kludge.
- **Paper-faithful TextEncoder/StyleEncoder/VF**:
  - `TextEncoderPaper(dim=128, attn_type='rope')` — paper A.2.2 (RoPE self-attn, 6 ConvNeXt + 4 attn + 2 cross-attn, shared 50×128 reference key)
  - `StyleEncoderTTLPaper(hdim=128, value_dim=128, out_scale=1.0)` — paper A.2.1 (no shipped-dist kludge)
  - `VectorField(dim=256, latent_dim=144, n_outer=4, inter=1024, learn_style_prototype=False)` — paper A.2.3, reference key shared from TextEncoder
- **DP paper-faithful path**: `StyleEncoderDPPaper(out_scale=1.0)` + `DurationPredictor(attn_type='rope')`. The estimator stays at the shipped 192-dim input because the paper text contradicts itself (says 64-dim ref + 164-dim estimator). `--paper_text_estimator` is an experimental flag using the inferred 128-dim layout.
- **CFG via UncondMasker (paper B.2)**: single joint Bernoulli with `p_uncond=0.05` per sample — replaces BOTH `text_emb` and `style_ttl` with learned uncond tokens together. Older 4%/1% split was abandoned.
- **Batch expander (K_e=4, paper Sec 3.2.3)**: after text/style encoders run once per sample, we duplicate conditioning 4× and pair each copy with a different (z_0, t). Per paper, more efficient than just bumping batch size for flow-matching convergence.
- **grad_accum** (`train_ttl.py`): microbatches per optimizer update. Paper effective batch 64 = `--batch_size 32 --grad_accum 2`. `--steps` counts optimizer updates so it matches paper iteration count.
- **5%-95% reference crop (DP, paper Sec 4.2)**: `sample_dp_reference_crop` picks crop length uniformly in `[0.05·T, 0.95·T]`, start uniformly in `[0, T-len]`. (Paper text is ambiguous on whether 5-95% means length range or start position; we picked length.)
- **σ_min=1e-8** (paper B.2). z_t = (1-(1-σ_min)t)z_0 + t·z_1, target = z_1 - (1-σ_min)z_0, L1 loss with reference mask m (1 outside crop, 0 inside).
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
