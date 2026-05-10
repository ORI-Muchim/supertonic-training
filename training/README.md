# Supertonic Training Code

This directory contains the current three-stage training implementation used in
this workspace:

```text
Stage 1: AE-GAN speech autoencoder
Stage 2: TTL text-to-latent flow matching
Stage 3: DP utterance-level duration predictor
```

The active goal is paper-faithful training on KSS for pronunciation and
alignment diagnostics. This is not expected to reproduce paper zero-shot quality
without paper-scale multi-speaker data.

## Current Local Artifacts

Keep these:

```text
training/runs/ae_paper_audit_crop1s_mean/
  ckpt_step01500000.pt
  cache/

training/runs/ttl_paper_rope_b32a2/
  config.json
  ckpt_step*.pt       # appears every 50k updates
  tb/
```

The old DP run was intentionally deleted because its checkpoint used the old
rel-pos path. Retrain DP after TTL finishes.

## Directory Map

```text
training/
  data/
    kss.py                 KSS index/dataset utilities
    spectrogram.py         228-mel feature extraction for AE
    ttl_dataset.py         cached latent dataset, TTL/DP ref crops and masks

  models/
    ae_encoder.py          paper AE encoder, 228 mel -> 24 latent
    ae_decoder.py          WaveNeXt-style decoder, 24 latent -> waveform
    ae.py                  SpecProcessor + encoder + decoder wrapper
    discriminators.py      MPD + MRD adversarial heads
    style_encoder.py       shipped and paper TTL/DP style encoders
    vector_field_cfg.py    CFG wrapper helpers

  losses/
    ae_losses.py           AE mel/adversarial/feature-matching losses
    flow_matching.py       TTL flow matching, CFG dropout, K_e expansion

  scripts/
    train_ae.py            Stage 1 AE-GAN trainer
    cache_latents.py       cache AE latents and mean/std stats
    train_ttl.py           Stage 2 paper TTL trainer
    train_dp.py            Stage 3 paper DP trainer
    synth_ae.py            AE-only reconstruction synth
    synth_ttl_paper.py     paper 128-dim TTL inference/synth
    eval_ae_recon.py       deterministic AE reconstruction metrics
    verify_decoder_weights.py
    verify_export_roundtrip.py
```

## Stage 1: AE

Local completed run:

```text
training/runs/ae_paper_audit_crop1s_mean/ckpt_step01500000.pt
```

Actual config:

```json
{
  "crop_seconds": 1.0,
  "adv_crop_seconds": 0.19,
  "spec_mode": "mel",
  "steps": 1500000,
  "batch_size": 16,
  "grad_clip": null,
  "enable_encoder_stem_bn": true,
  "enable_encoder_out_ln": true,
  "enable_decoder_stem_bn": true,
  "lrecon_reduction": "mean"
}
```

Notes:

- The paper specifies 228 mel bands, not the old 1253 mel+STFT concat path.
- The paper clearly specifies the adversarial crop duration of 0.19s. The full
  reconstruction path uses 1.0s crops locally for better AE training stability.
- The paper does not explicitly state whether the multi-resolution mel terms are
  summed or averaged. This repo uses mean-scale reduction after audit so the
  lambda value is not silently multiplied by the number of resolutions.
- AE-only reconstruction is much closer to real mel than end-to-end TTL output:
  measured log-mel MAE was about 0.51 for real vs AE-only, versus about 3.27
  for real vs E2E TTL. Current pronunciation/quality bottleneck is therefore
  mainly TTL, not AE.

Re-run command:

```powershell
python -m training.scripts.train_ae `
  --batch_size 16 `
  --steps 1500000 `
  --out_dir training/runs/ae_paper_audit_crop1s_mean
```

Cache latents:

```powershell
python -m training.scripts.cache_latents `
  --ckpt training/runs/ae_paper_audit_crop1s_mean/ckpt_step01500000.pt `
  --out_dir training/runs/ae_paper_audit_crop1s_mean/cache
```

## Stage 2: TTL

Current active run:

```text
training/runs/ttl_paper_rope_b32a2
```

Actual config:

```json
{
  "paper_faithful": true,
  "fine_tune": false,
  "lr": 0.0005,
  "beta1": 0.9,
  "beta2": 0.999,
  "weight_decay": 0.0,
  "grad_clip": null,
  "steps": 700000,
  "lr_halve_every": 300000,
  "batch_size": 32,
  "k_expand": 4,
  "grad_accum": 2,
  "attn_type": "rope",
  "prob_uncond": 0.05,
  "ckpt_every": 50000
}
```

Paper-faithful TTL choices:

- `TextEncoderPaper(dim=128, attn_type="rope")`
- `StyleEncoderTTLPaper(hdim=128, value_dim=128, out_scale=1.0)`
- `VectorField(dim=256, latent_dim=144, n_outer=4, inter=1024)`
- TextEncoder and VectorField share the same 50x128 reference key.
- `sigma_min=1e-8`
- `p_uncond=0.05`, one joint dropout event for both text and style.
- `K_e=4`
- Reference crop: 0.2s to 9s, capped at half the utterance.
- Loss mask `m`: 1 outside the reference crop, 0 inside the reference crop.
- Optimizer updates count as `steps`; microbatch accumulation does not change
  the paper iteration count.

Launch command:

```powershell
python -m training.scripts.train_ttl `
  --cache_dir training/runs/ae_paper_audit_crop1s_mean/cache `
  --batch_size 32 `
  --grad_accum 2 `
  --attn_type rope `
  --num_workers 0 `
  --out_dir training/runs/ttl_paper_rope_b32a2
```

Why `batch=32, grad_accum=2`:

- Effective batch is 64, matching the paper interpretation used here.
- `batch=64, accum=1` was benchmarked and was slower because variable-length
  padding made the microbatch too expensive.
- `batch=32, accum=2` keeps GPU utilization high and runs around 2.5 updates/s
  on the local RTX 3090.

Monitor:

```powershell
Get-Content training/runs/ttl_paper_rope_b32a2.log -Tail 10
nvidia-smi
```

## Stage 3: DP

DP should be trained after TTL completes.

Current code defaults:

- `paper_faithful=true`
- `StyleEncoderDPPaper(out_scale=1.0)`
- `DurationPredictor(attn_type="rope")`
- Batch size 128
- 3000 steps
- AdamW lr `5e-4`
- No grad clipping by default
- Reference crop uses a 5 percent to 95 percent random length crop.

Launch after TTL:

```powershell
python -m training.scripts.train_dp `
  --cache_dir training/runs/ae_paper_audit_crop1s_mean/cache `
  --num_workers 0 `
  --out_dir training/runs/dp_paper_rope_3k
```

DP ambiguity:

- The paper text says the reference embedding is 64-dim but also mentions a
  164-dim estimator layer, which is internally inconsistent with the rest of the
  dimensions.
- The default implementation keeps the released ONNX/deployed estimator input
  shape: 64 text + 8*16 reference = 192.
- `--paper_text_estimator` exists only for experiments and is not the default.

## Mid-Training Synthesis

Use `synth_ttl_paper.py` for current paper 128-dim TTL checkpoints:

```powershell
python -m training.scripts.synth_ttl_paper `
  --ae_ckpt training/runs/ae_paper_audit_crop1s_mean/ckpt_step01500000.pt `
  --cache_dir training/runs/ae_paper_audit_crop1s_mean/cache `
  --ttl_ckpt training/runs/ttl_paper_rope_b32a2/ckpt_step00050000.pt `
  --text "안녕하세요 슈퍼토닉 학습 테스트입니다." `
  --cfg_scale 3.0 `
  --steps 5 `
  --out out_ttl_50k.wav
```

After DP exists, pass `--dp_ckpt` to use predicted duration. Before DP exists,
use explicit or heuristic duration.

## Diagnostics

AE-only reconstruction:

```powershell
python -m training.scripts.synth_ae `
  --ckpt training/runs/ae_paper_audit_crop1s_mean/ckpt_step01500000.pt `
  --out_dir training/runs/ae_paper_audit_crop1s_mean/synth_diag
```

Mel comparison:

```powershell
python analysis/mel_compare.py
```

Expected interpretation:

- If AE-only sounds clean but TTL E2E sounds muddy, the bottleneck is TTL.
- If AE-only is already bad, the bottleneck is AE.
- Current measurements indicated TTL is the main bottleneck: real vs AE-only
  log-mel MAE was about 0.51, while real vs E2E TTL was about 3.27.

## Cleanup Policy

Safe to delete:

- smoke runs
- benchmark runs
- old rel-pos DP runs
- old TTL runs superseded by `ttl_paper_rope_b32a2`
- AE intermediate checkpoints if the final 1.5M checkpoint and cache are kept

Do not delete while training:

- `training/runs/ttl_paper_rope_b32a2`
- `training/runs/ae_paper_audit_crop1s_mean/cache`
- `training/runs/ae_paper_audit_crop1s_mean/ckpt_step01500000.pt`

## Zero-Shot Reality Check

The model architecture is designed for zero-shot voice cloning, but the data
must contain many speakers.

KSS-only training cannot learn zero-shot speaker conditioning because every
reference clip belongs to the same speaker. To train zero-shot behavior, AE,
TTL, and DP all need multi-speaker speech/text data.
