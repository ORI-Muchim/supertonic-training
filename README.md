# Supertonic Training Pipeline

Reverse-engineered training and inference code for SupertonicTTS
(paper: arXiv 2503.23108), built from the released ONNX models and the paper.

This is not an official Supertone repository. No shipped weights are
redistributed here. The original model, paper, and released assets belong to
Supertone.

## Current Local Status

As of 2026-05-11, this workspace is focused on a KSS single-speaker
paper-faithful reproduction, not on shipped-weight voice cloning.

Completed:

- Stage 1 AE trained to 1.5M steps on KSS.
- AE final checkpoint:
  `training/runs/ae_paper_audit_crop1s_mean/ckpt_step01500000.pt`
- AE latent cache:
  `training/runs/ae_paper_audit_crop1s_mean/cache`

Running:

- Stage 2 TTL paper-faithful rerun:
  `training/runs/ttl_paper_rope_b32a2`
- Settings: `batch_size=32`, `grad_accum=2`, effective batch 64,
  `attn_type=rope`, `K_e=4`, `sigma_min=1e-8`, `p_uncond=0.05`.

Pending:

- Stage 3 DP must be retrained after TTL finishes.
- The old DP checkpoint was removed because it was trained with the old rel-pos
  path, not the current RoPE paper path.

## What This Repo Contains

Two layers are maintained:

1. `analysis/`
   PyTorch reimplementations of the released ONNX inference modules:
   `duration_predictor`, `text_encoder`, `vector_estimator`, and `vocoder`.
   These are used both for verification and as the training-time forward
   modules where appropriate.

2. `training/`
   Three-stage training code:
   AE-GAN -> TTL flow matching -> DP duration prediction.

The current training work is in `training/`. See `training/README.md` for the
detailed recipe and command list.

## Important Scope

The architecture and training recipe are implemented to match the paper where
the paper is explicit. This does not mean the local KSS run will match the
paper's zero-shot quality.

The paper used much larger data:

- AE: 11,167 hours, about 14,000 speakers.
- TTL/DP: 945 hours, about 2,576 speakers.

The local run uses:

- KSS: 12.86 hours, 1 Korean female speaker.
- One RTX 3090.

Therefore:

- Single-speaker KSS TTS can be studied and improved here.
- Zero-shot voice cloning requires multi-speaker AE + TTL + DP training data.
- KSS-only training cannot teach the model to use reference speaker variation.

## Paper-Faithful Definition Used Here

The current Stage 2 TTL path matches the paper implementation decisions used in
this repo:

- TextEncoderPaper: dim 128, RoPE self-attention.
- StyleEncoderTTLPaper: 50 style tokens, value dim 128, output scale 1.0.
- VectorField: dim 256, latent dim 144, `K_e=4`.
- TextEncoder and VectorField share the same 50x128 reference key.
- Flow matching uses `sigma_min=1e-8`.
- Classifier-free dropout is one joint Bernoulli event, `p_uncond=0.05`.
- Reference crop is 0.2s to 9s and no more than half the utterance.
- Flow loss mask excludes the reference crop region.
- AdamW uses lr `5e-4`, halved every 300k TTL updates.
- Current single-3090 effective batch is 64 via `batch_size=32` and
  `grad_accum=2`.

Known paper ambiguities or unavoidable differences:

- The paper does not publish the official training code.
- RoPE convention is not specified down to implementation details.
- AE multi-resolution reconstruction reduction is ambiguous; this repo uses
  mean-scale reduction after audit.
- DP estimator dimensions in the paper text are internally inconsistent; the
  default follows the released ONNX/deployed 192-dim estimator shape while using
  paper RoPE and paper style encoder scaling.
- Data, speaker count, language distribution, and GPU scale differ heavily from
  the paper.

## Setup

```powershell
conda create -n supertonic python=3.11 -y
conda activate supertonic
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download the official Supertonic release assets separately and place them under:

```text
assets/
  onnx/
    duration_predictor.onnx
    text_encoder.onnx
    vector_estimator.onnx
    vocoder.onnx
  tts.json
  unicode_indexer.json
  voice_styles/
    F1.json ... F5.json
    M1.json ... M5.json
```

## Verification

```powershell
# End-to-end PyTorch vs ONNX verification
python analysis/verify_end_to_end.py

# AE decoder vs released vocoder decoder
python -m training.scripts.verify_decoder_weights

# Export round-trip checks
python -m training.scripts.verify_export_roundtrip
```

## Paper-Faithful KSS Training Path

Stage 1 AE was already run locally:

```powershell
python -m training.scripts.train_ae `
  --batch_size 16 `
  --steps 1500000 `
  --out_dir training/runs/ae_paper_audit_crop1s_mean
```

Cache AE latents:

```powershell
python -m training.scripts.cache_latents `
  --ckpt training/runs/ae_paper_audit_crop1s_mean/ckpt_step01500000.pt `
  --out_dir training/runs/ae_paper_audit_crop1s_mean/cache
```

Current Stage 2 TTL run:

```powershell
python -m training.scripts.train_ttl `
  --cache_dir training/runs/ae_paper_audit_crop1s_mean/cache `
  --batch_size 32 `
  --grad_accum 2 `
  --attn_type rope `
  --num_workers 0 `
  --out_dir training/runs/ttl_paper_rope_b32a2
```

After TTL completes, retrain DP with the current paper/RoPE code:

```powershell
python -m training.scripts.train_dp `
  --cache_dir training/runs/ae_paper_audit_crop1s_mean/cache `
  --num_workers 0 `
  --out_dir training/runs/dp_paper_rope_3k
```

## Monitoring The Current Run

```powershell
Get-Content training/runs/ttl_paper_rope_b32a2.log -Tail 10

Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*ttl_paper_rope_b32a2*' } |
  Select-Object ProcessId, CommandLine

nvidia-smi
```

Expected healthy TTL signs:

- Loss should trend downward over tens of thousands of steps.
- Grad norm should stay finite and small.
- GPU memory around 14GB to 18GB on the current `batch=32, accum=2` run.
- First checkpoint is saved at 50k updates.

## Inference During Development

For paper-faithful 128-dim TTL checkpoints, use:

```powershell
python -m training.scripts.synth_ttl_paper `
  --ae_ckpt training/runs/ae_paper_audit_crop1s_mean/ckpt_step01500000.pt `
  --cache_dir training/runs/ae_paper_audit_crop1s_mean/cache `
  --ttl_ckpt training/runs/ttl_paper_rope_b32a2/ckpt_step00050000.pt `
  --text "안녕하세요 슈퍼토닉 학습 테스트입니다." `
  --cfg_scale 3.0 `
  --out out_ttl_test.wav
```

Use `--dp_ckpt` after the DP has been retrained.

## Notes On Fine-Tuning Shipped Weights

Older README text described a shipped-weight fine-tune path. That path is still
useful for experiments, but it is not the current paper-faithful KSS
from-scratch path.

The current priority is:

1. Finish TTL with RoPE + effective batch 64.
2. Retrain DP with RoPE.
3. Evaluate same-text and new-text pronunciation.
4. Decide whether larger multi-speaker data is needed.

## License And Credits

Original architecture, released model assets, and paper are by Supertone.
This repository contains reverse-engineering notes and training scripts for
research and educational use.

- Paper: arXiv 2503.23108
- Official code: https://github.com/supertone-inc/supertonic
- Released weights: https://huggingface.co/Supertone/supertonic-2
