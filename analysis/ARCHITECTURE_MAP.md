# Supertonic-2 Architecture Map (reverse-engineered from ONNX weights)

Source: `assets/onnx/{duration_predictor, text_encoder, vector_estimator, vocoder}.onnx`,
plus `tts.json` and `unicode_indexer.json`.

**Total params: 65,498,438 (65.5 M, README says "66M" ✓)**

| Model | Params | File | Real tensors |
|---|---:|---:|---:|
| duration_predictor | 0.34 M | 1.52 MB | 98 |
| text_encoder | 6.77 M | 27.43 MB | 141 |
| vector_estimator | 33.01 M | 132.47 MB | 349 |
| vocoder (= AE decoder) | 25.34 M | 101.41 MB | 103 |

All real weights exported with PyTorch names under `tts.ae.*`, `tts.ttl.*`, `tts.dp.*`.

---

## 0. Shared ConvNeXt-1D Block

Every ConvNeXt block in the whole system has the same structure (GELU activation = Erf op confirms):

```
x → DepthwiseConv1d(ksz=5 or 7, dilation=d)
  → LayerNorm (channels-last)
  → PointwiseConv1d(dim → intermediate)
  → GELU
  → PointwiseConv1d(intermediate → dim)
  → γ · (learnable [1, dim, 1])
  → + residual
```

Weight names per layer: `dwconv.weight/bias`, `norm.norm.weight/bias`, `pwconv1.weight/bias`, `pwconv2.weight/bias`, `gamma`.

Vocoder uses `dwconv.net.weight/bias` (Sequential wrapper, ksz=7), others use `dwconv.weight` (ksz=5).

---

## 1. Text Input & Char Vocabulary

- `unicode_indexer.json`: **list[65536]** mapping each BMP codepoint → int
  - Valid ids: 0..161 (162 real chars covering en/ko/es/pt/fr + punctuation + `<lang>` tags)
  - `-1` = unknown; PyTorch embedding(-1) → last row (idx 162)
- `char_embedder.weight [163, dim]` consistently across models (dim 64 for DP, 256 for TTL text encoder)
- Preprocess (from `py/helper.py`): NFKD normalize → strip emoji/special → wrap `<{lang}>...</{lang}>` → codepoint → indexer

---

## 2. `vocoder.onnx` — AE Decoder + 6× un-chunker

**Input:** `latent [B, 144, L]` (TTL latent, already un-normalized scale 0.25)
**Output:** `wav_tts [B, L·512·6/base_chunk]` — waveform

Weights (103 tensors, 25.34 M):

| Module | Shape / note |
|---|---|
| `tts.ae.latent_mean [1,24,1]` | AE latent mean (de-normalize before decode) |
| `tts.ae.latent_std [1,24,1]` | AE latent std |
| `tts.ae.decoder.convnext.{0..9}.*` | 10× ConvNeXt (hdim=512, inter=2048, ksz=7). Dilations per `tts.json`: `[1,2,4,1,2,4,1,1,1,1]` |
| `tts.ae.decoder.final_norm.norm.*` | BatchNorm1d (512) — weight/bias/running_mean/running_var |
| `tts.ae.decoder.head.layer1.net.weight [2048,512,3]` | Conv1d 512→2048, ksz=3 |
| `tts.ae.decoder.head.layer2.weight [512,2048,1]` | Conv1d 2048→512 |
| trailing `PRelu` (onnx::PRelu_1505) | final activation on waveform samples |

Architecture flow:
```
latent[B,144,L]
  → reshape [B,24,L*6]              # un-chunk (chunk_compress_factor=6)
  → * latent_std + latent_mean      # de-normalize to AE latent scale
  → stem Conv1d 24→512 (ksz=7)      # onnx::Conv_1440/1441 [512,24,7]
  → 10× ConvNeXt (dilations above)
  → final_norm (BatchNorm1d 512)
  → head: Conv1d 512→2048 ksz=3 → GELU → Conv1d 2048→512 ksz=1 → PReLU
  → reshape to waveform [B, L*6*chunk_per_step]
```

**Note:** The `head` 2-stage design (512→2048→512) matches `tts.json ae.decoder.head` (`idim=512, hdim=2048, odim=512, ksz=3`). The final reshape converts channel dim to samples (each channel predicts a sample within a chunk).

---

## 3. `duration_predictor.onnx` — Utterance-level Duration

**Input:** `text_ids [B, T]`, `style_dp [B, 8, 16]`, `text_mask [B, 1, T]`
**Output:** `duration [B]` (scalar seconds per sample)

Two sub-modules:

### 3a. sentence_encoder → 64-dim utterance embedding
- `text_embedder.char_embedder.weight [163, 64]`
- `sentence_token [1, 64, 1]` — prepended CLS-like learnable token
- 6× ConvNeXt (hdim=64, inter=256, ksz=5, dilations all 1)
- 2× Attention encoder block (relative-position self-attn):
  - `conv_q/k/v/o.weight [64,64,1]` — implemented as 1×1 Conv1d
  - `emb_rel_k [1,9,32]`, `emb_rel_v [1,9,32]` — **relative position embeddings** (window=4, head_dim=32, 2 heads). Note: DP uses relative-pos attention, NOT RoPE
  - `ffn_layers.{i}.conv_1 [256,64,1]`, `conv_2 [64,256,1]`
  - `norm_layers_1/2` — pre-norm
- `proj_out.net.weight [64,64,1]` — final 1×1 conv
- Read `sentence_token` position → 64-dim utterance vector

### 3b. predictor MLP
- Input: concat(sentence_vec[64], style_dp_flat[8*16=128]) → 192
- `layers.0 [128, 192]` + PReLU(1 shared channel) + `layers.1 [1, 128]`
- Output: scalar duration (seconds)
- Final ops: `Exp` then `Clip` → positivity + bound. Matches config `normalizer.scale=1.0`.

---

## 4. `text_encoder.onnx` — Style-conditioned Text Encoder

**Input:** `text_ids [B, T]`, `style_ttl [B, 50, 256]`, `text_mask [B, 1, T]`
**Output:** `text_emb [B, 256, T]`

### 4a. text_encoder (main stack, 6.37 M)
- `char_embedder.weight [163, 256]`
- 6× ConvNeXt (hdim=256, inter=1024, ksz=5, dilations all 1)  — confirms `tts.json ttl.text_encoder.convnext`
- 4× Attention encoder block with relative-position self-attention:
  - Weights: `attn_layers.{i}.conv_q/k/v/o [256,256,1]`, `emb_rel_k/v [1, 9, 64]` (4 heads × head_dim 64)
  - FFN: `conv_1 [1024,256,1]`, `conv_2 [256,1024,1]`
  - **Relative-pos window = 4** (9 = 2·4+1)
- `proj_out`: Conv1d 256→256, ksz=1

### 4b. speech_prompted_text_encoder (2.0 K params, cross-attn to style)
- Two attention layers; each has: `W_query`, `W_key`, `W_value`, `out_fc` (all Linear → 256 dim)
- Cross-attn: **Q = text features (256), K/V = style_ttl [50, 256]**
- Injects voice-style into text conditioning
- Note: attention weights (`W_*.linear.weight`) are stored as **ONNX `onnx::MatMul_*` constants** (not under `tts.*` names) because `nn.Linear` weights collapsed into MatMul during export
  - Six `[256,256]` MatMul consts per cross-attn pair → `Q, K, V, out` × 2 = 8 matrices, but only 6 visible (some may be fused)

Total text_encoder params sum:
- tts.ttl.* named: 6,767,872
- + 6× `MatMul_36xx [256,256]` constants (392,192)
- = ~7.16 M (close to model params 6.79 M; remaining are bias + small norms)

---

## 5. `vector_estimator.onnx` — Flow-Matching Vector Field (33 M, the workhorse)

**Input:** `noisy_latent [B, 144, L]`, `text_emb [B, 256, T]`, `style_ttl [B, 50, 256]`,
`latent_mask [B, 1, L]`, `text_mask [B, 1, T]`, `current_step [B]`, `total_step [B]`
**Output:** `denoised_latent [B, 144, L]`

Step fraction `t = current_step / total_step` is computed inside the graph (matches `Div` ops).

### 5a. Global (not in main_blocks)
| Tensor | Shape | Role |
|---|---|---|
| `tts.ttl.normalizer.scale` | scalar (0.25 per config) | latent pre/post scale |
| `tts.ttl.vector_field.proj_in.net.weight` | `[512, 144, 1]` | Conv1d 144→512 entry |
| `tts.ttl.vector_field.proj_out.net.weight` | `[144, 512, 1]` | Conv1d 512→144 exit |
| `tts.ttl.vector_field.time_encoder.mlp.0.linear.weight` | `[256, 64]` | time MLP layer 1 |
| `tts.ttl.vector_field.time_encoder.mlp.1.linear.weight` | `[64, 256]` (via MatMul const) | time MLP layer 2 (→ 64-dim time emb) |

### 5b. `main_blocks` — **24 entries = 4 repeats × 6 sub-modules**

Confirmed by looking at which tensor names belong to each index:

| Idx (mod 6) | Module | Named bias tensors | Matching `tts.json` field |
|---|---|---|---|
| 0 | `convnext` 4-layer | `convnext.0..3.{dwconv,norm,pwconv1,pwconv2,gamma}` (dilations **[1,2,4,8]**) | `convnext_0` |
| 1 | `linear.linear` (FiLM time) | `linear.linear.bias [512]` (weight is `onnx::MatMul_309x [64,512]`) | `time_cond_layer (idim=512, time_dim=64)` |
| 2 | `convnext` 1-layer | `convnext.0.*` (dilation 1) | `convnext_1` |
| 3 | `attn` (LARoPE cross-attn) | `attn.{W_q,W_k,W_v,out_fc}.linear.bias`, `attn.theta [1,1,32]`, `attn.increments [1,1000,1]`, `norm` | **`text_cond_layer`** (4 heads, head_dim=64, rotary_base=10000, rotary_scale=10, `use_residual: true`) |
| 4 | `convnext` 1-layer | `convnext.0.*` | `convnext_2` |
| 5 | `attention` (cross-attn to style) | `attention.{W_q,W_k,W_v,out_fc}.linear.bias`, `norm` | `style_cond_layer` (no positional enc; K/V from 50 style tokens) |

So the actual forward order is: `convnext_0 → time_FiLM → convnext_1 → text_CA(LARoPE) → convnext_2 → style_CA`, repeated 4×.

**LARoPE implementation details (from theta/increments):**
- `theta [1, 1, 32]` stores the 32 per-dim inverse-frequency constants `θ_j = rotary_base^(-2j/d)` with d=64
- `increments [1, 1000, 1]` stores precomputed position indices 0..999 (max L=1000)
- Angle = `rotary_scale · (p/L) · θ_j` with `rotary_scale=10`, `L` = seq length at runtime
- Explains why only 3 `Sin/Cos` pairs in the graph: **the 4 main_blocks share computed cos/sin tables** (plus 1 for time sinusoidal embed). ONNX CSE collapsed duplicates.

### 5c. `last_convnext` — 4-layer ConvNeXt (dilations [1,1,1,1]), before proj_out.

### 5d. Param budget for vector_estimator
- Named `tts.ttl.vector_field.*`: ~29.77 M
  - main_blocks: 25.33 M (each full block ≈ 6.33 M; breakdown: convnext_0=4.22M, time_FiLM+bias=0.5K, convnext_1=1.05M, text_CA=3K named (rest in MatMul), convnext_2=1.05M, style_CA=2K named (rest in MatMul))
  - last_convnext: 4.22 M
  - proj_in/proj_out: 0.15 M
  - time_encoder.mlp named: 33 K
- `onnx::MatMul_*` constants (attention Q/K/V/out + time MLP): ~3.24 M
- Total 33.01 M ✓

---

## 6. Voice Style JSON (`voice_styles/*.json`)

Two tensors per speaker, precomputed from a reference utterance:

| Key | Shape | Purpose |
|---|---|---|
| `style_ttl` | `[1, 50, 256]` | 50 learnable-token outputs of TTL style encoder (GST variant, used by text_encoder + vector_estimator) |
| `style_dp` | `[1, 8, 16]` | 8 × 16 style tokens for duration predictor |

These are the *outputs* of style encoders; style encoders themselves are NOT in the shipped ONNX. To train from scratch you need to implement the **style encoders** (convnext + style_token_layer in `tts.json.ttl.style_encoder` and `tts.json.dp.style_encoder`) and run them during training.

---

## 7. What's NOT in ONNX (must be re-implemented for training)

1. **AE encoder** (`tts.ae.encoder` in config):
   - MelSpectrogram(228 mel, n_fft=2048, hop=512, win=2048, SR=44.1k)
   - Conv stem 1253→512 (ksz=7) — NOTE: `idim=1253` is unusual; likely mel+extras (229+ mels?). See `tts.json ae.encoder.idim=1253`. Could be 228 mels × 5 frames concat or similar.
   - 10× ConvNeXt (ksz=7, all dilations 1, hdim=512, inter=2048)
   - Project 512→24 (latent)
2. **Discriminators** (multi-period {2,3,5,7,11}, multi-resolution FFT {512,1024,2048})
3. **Style encoders** (TTL and DP) — train-time: extract style from ref mel; ship as frozen JSON at inference
4. **`batch_expander`** (K_e=6 from config) — duplicate condition encodings across 6 perturbed `(z_0, t)` pairs
5. **`uncond_masker`** (CFG + SPFM dropping) — `prob_both_uncond=0.04`, `prob_text_uncond=0.01`, std=0.1 noise

---

## 8. Sanity-check Exercise (reconstruct PyTorch → match outputs)

Minimal verification loop:
1. Build PyTorch `VectorField` with exact shapes from `tts.json`
2. Load weights from ONNX: `onnx.numpy_helper.to_array(init)` → assign to `module.state_dict()`
   - For `onnx::MatMul_*` constants, need a mapping table (since PyTorch `Linear.weight` is `[out, in]` but MatMul is `[in, out]` → transpose)
3. Run dummy `(text_ids, style_ttl, noisy_latent)` through both PyTorch and ONNXRuntime
4. Check `max|y_torch - y_onnx| < 1e-4`

Per-module dev order (easy → hard):
1. vocoder (pure conv, no attention) — easiest
2. duration_predictor (small, rel-pos attn)
3. text_encoder (rel-pos attn + style cross-attn)
4. vector_estimator (FiLM + LARoPE + 2 cross-attns) — hardest

---

## 9. Training Pipeline (stage recipe)

Matches paper 2503.23108:

```
Stage 1: AE (GAN) — 1.5M steps, AdamW LR=2e-4, batch 128, 4×4090
  L_G = 45·L_mel(MS-L1, FFT ∈ {1024,2048,4096}) + 1·L_adv + 0.1·L_fm
  Discriminators: MPD {2,3,5,7,11} + MRD {512,1024,2048}
  + compute latent_mean/std over train set, save into vocoder checkpoint

Stage 2: TTL flow matching — 700k steps, AdamW LR=5e-4 (halve every 300k), batch 64 × K_e=6
  CFM loss: L1(v_θ(z_t, cond, t) − (z_1 − (1−σ_min)·z_0)) with σ_min≈0
  + uncond_masker: p(both_uncond)=0.04, p(text_uncond)=0.01, Gaussian noise std=0.1
  + SPFM filter (after warmup, t'=0.5): if L_cond > L_uncond → use L_uncond that sample

Stage 3: DP — 3k steps, AdamW LR=5e-4, batch 128, 1×4090
  L = L1(pred_dur_sec, gt_dur_sec)  (utterance-level scalar)
```

At inference: Euler ODE solver, `total_step ∈ {2,5}`, optional CFG=3.
