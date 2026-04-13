# Supertonic-2 리버스 엔지니어링 핸드오프

Supertonic-2의 추론 ONNX 4개를 PyTorch로 역설계해 **훈련 코드 작성 기반**을 만들어둔 상태입니다.
총 65.5M 파라미터 전부 매핑 완료, 출력은 ONNX와 수치적으로 일치합니다.

## 1. 레포 구조

```
supertonic/
├── assets/onnx/             # HuggingFace에서 clone (git-lfs 필요, 263MB)
│   ├── duration_predictor.onnx
│   ├── text_encoder.onnx
│   ├── vector_estimator.onnx
│   ├── vocoder.onnx
│   ├── tts.json             # 모든 레이어 설정이 여기 있음
│   └── unicode_indexer.json # BMP codepoint → token id (vocab=163)
├── py/                      # 기존 공식 ONNX Runtime 추론 예제
└── analysis/                # 우리 작업 결과물
    ├── ARCHITECTURE_MAP.md           # ONNX ↔ tts.json 전체 매핑 문서
    ├── HANDOFF.md                    # 이 파일
    ├── dump_onnx.py                  # ONNX → init/node TSV 덤프
    ├── summarize_params.py           # 파라미터 prefix별 그룹핑
    ├── dump/                         # 덤프 결과 (inits.tsv, nodes.tsv, summary.txt)
    ├── torch_vocoder.py              # ✅ AE 디코더 (25.3M, 오차 1.68e-6)
    ├── torch_duration_predictor.py   # ✅ 길이 예측 (0.34M, 3.58e-7)
    ├── torch_text_encoder.py         # ✅ 텍스트 인코더 (6.77M, 2.52e-5)
    ├── torch_vector_estimator.py     # ✅ Flow matching 필드 (33.0M, 2.44e-6)
    ├── debug_*.py                    # 모듈별 stage-by-stage 디버거
    └── verify_end_to_end.py          # 전체 파이프라인 PyTorch vs ONNX 비교
```

## 2. 아키텍처 요약

```
text ──► [duration_predictor] ─► 발화 길이 (초, 스칼라 1개)
     │
     ├─► [text_encoder] ─► text_emb [B, 256, T]
     │        └─ 6× ConvNeXt(256) → 4× attn(rel-pos, 4H×64D)
     │        └─ speech_prompted: 2× cross-attn (prototype K + W_V(style))
     │
     └─► noisy_latent z_0 ~ N(0, I) sampled from duration
                │
                ▼
          [vector_estimator] × N steps (Euler ODE)
                │   per step: 4 outer blocks, 각 block = convnext_0(4L) → time_FiLM
                │             → convnext_1(1L) → text_CA(LARoPE) → convnext_2(1L)
                │             → style_CA(prototype K + W_V(style))
                │   출력: x_{t+1} = x_t + (1/N) · v_θ
                ▼
          [vocoder] (= AE decoder)
                └─ un-chunk 6× → de-normalize → 10× ConvNeXt
                   → BatchNorm → head(512→2048→512) → PReLU → [B, T·3072] waveform (44.1kHz)
```

학습 스테이지 (논문 arXiv 2503.23108, 4×RTX4090 기준):
1. **AE (GAN)**: AdamW 2e-4, batch 128, 1.5M step — 45·L_mel + 1·L_adv + 0.1·L_fm
2. **TTL (flow matching)**: AdamW 5e-4 (300k마다 반감), batch 64 × K_e=6, 700k step
3. **DP**: AdamW 5e-4, batch 128, 3k step — L1 on total duration (scalar)

## 3. 리버스 엔지니어링에서 발견한 비자명한 디테일들

검증 과정에서 버그로 드러나 고친 것들 — **재구현할 때 반드시 주의**:

### 패딩
| 모듈 | 모드 | 방향 |
|---|---|---|
| vocoder ConvNeXt (ksz=7) | `replicate` | **causal** (left only, `pad=(k-1)·d`) |
| 그 외 모든 ConvNeXt (ksz=5) | `replicate` | **symmetric** (양쪽 `pad=(k-1)·d/2`) |

zero-pad이 아니라 **replicate/edge pad**라는 점이 큰 차이를 만듭니다.

### Time 임베딩
- 주파수: `freqs[j] = 10000^(-j/(half-1))` — **표준 RoPE의 `-2j/dim`이 아님**
- Sinusoidal 이전에 **`t *= 1000`** 스케일
- MLP 활성함수: **Mish** (`x · tanh(softplus(x))`)
- FiLM은 shift-only (scale 없음, 단일 Linear 64→512)

### LARoPE (text cross-attn, main_blocks.3/9/15/21)
- `θ_j = 10 · 10000^(-j/31)` — **rotary_scale γ=10이 theta에 baked-in**
- 각도 = `(p/L) · θ` (L은 해당 시퀀스 actual 길이, `ReduceSum(mask)`로 계산)
- RoPE 회전 방식: **halves concat** (LLaMA식) — `(x1·cos − x2·sin, x1·sin + x2·cos)` where `x = [x1, x2]` 절반
- `theta [1,1,32]`와 `increments [1,1000,1]`은 **4개 블록이 공유** — main_blocks.3에만 저장됨

### Attention 일반
- Scale = **`sqrt(attn_dim=256) = 16`**, **`sqrt(head_dim)` 아님**
- Mask fill: softmax 전 `-1e4`, 또는 softmax 후 query 위치에 `* mask_q` 곱 (둘 다 존재)
- VITS rel-pos attention의 `rel_to_abs` 트릭: F.pad 기반 구현이 미묘하게 깨지기 쉬우니 **직접 gather**가 안전 (DP/text_encoder)

### Style cross-attn (main_blocks.5/11/17/23)
- **K source는 style_ttl이 아니라 학습된 prototype `[1, 50, 256]`** (모든 블록 공유, ONNX에서 `/Expand_output_0` initializer)
- K = `tanh(W_key(prototype))` — runtime Tanh
- V만 `W_value(style_ttl)`
- 2 heads × 128 head_dim

### Text encoder
- `proj_out`가 config엔 있지만 export시 생략됨 (있다고 로드하면 KeyError)
- speech_prompted 내부는 특이한 residual: `q1 = x + attn1(x)`, 최종 출력은 `x + attn2(q1)` (attn1 결과는 attn2의 Q에만 영향, 최종합에선 사라짐)
- speech_prompted K는 prototype이라 W_key 없음 (3개 Linear: W_q, W_v, out_fc)

### Vector estimator 최종 출력
- 모델 출력 = **Euler step 결과** (velocity 아님): `denoised = noisy_latent + (1/total_step) · v_θ`
- 추론 시 여러 step을 돌릴 때 ONNX 한 번 호출 = ODE 한 step

### ONNX initializer 매핑
- PyTorch `nn.Linear.weight`는 `[out, in]`이지만 ONNX의 `onnx::MatMul_*` 상수는 `[in, out]` — **transpose 필요**
- PReLU shape [1,1]로 저장되어 있어 `num_parameters=1`에는 `.reshape(-1)` 해줘야 함
- BatchNorm1d running stats 4개(weight/bias/running_mean/running_var) 모두 로드 필요

## 4. 검증 결과

각 모듈을 단독으로 돌려 ONNX Runtime 출력과 비교한 max abs diff:
- `vocoder`: **1.68e-6**
- `duration_predictor`: **3.58e-7**
- `text_encoder`: **2.52e-5**
- `vector_estimator`: **2.44e-6**

전체 파이프라인(텍스트→발화) 4개 언어·화자 × 5 ODE step 누적:
- wav max diff **6.3e-5 ~ 3.0e-3** (신호 범위 ±0.3 기준)
- wav mean diff ~5e-6 — 들리지 않는 수준 (softmax 집중점 근처 FP roundoff로 발생하는 피크성 오차)

모든 검증은 `python analysis/verify_end_to_end.py`로 재현 가능.

## 5. 지금 할 수 있는 것

**Forward는 ONNX와 bitwise-close**이므로:
- 훈련된 가중치로 **fine-tuning** (실용적)
- 같은 구조로 **from-scratch 훈련** 시 loss만 붙이면 됨 (훈련 레시피는 논문 재현)
- 이 PyTorch 모듈을 그대로 export하면 **기존 ONNX 런타임과 호환 유지**

## 6. 훈련 코드 작성 시 아직 없는 것 (새로 짜야 함)

| 구성요소 | 왜 없나 | 복잡도 |
|---|---|---|
| **AE encoder** (mel 228 → latent 24) | 추론엔 디코더만 필요 | 낮음 (ConvNeXt 10L, 대칭 구조) |
| **Style encoder × 2** (TTL GST 50-token, DP 8-token) | 추론엔 미리 뽑은 JSON 사용 | 중간 (style_token_layer, tts.json에 설정 있음) |
| **Discriminators** (MPD {2,3,5,7,11} + MRD {512,1024,2048}) | 추론 무관 | 낮음 (HiFi-GAN 표준) |
| **손실함수** | — | 낮음 (L1, multi-scale mel, feature matching) |
| **SPFM / batch_expander / uncond_masker** | 학습 전용 | 중간 |
| **Vector estimator의 velocity forward** | 현재 Euler step 섞여 있음 | 낮음 (`proj_out` 직후 분기) |

**주의사항**:
- `tts.json ae.encoder.idim=1253` — 이게 정확히 뭔지 확인 필요 (228 mel × N프레임 concat? 아직 미확인)
- Text normalizer 학습 레시피는 공개 안 됨 (논문이 주장하는 "숫자·날짜 자연 처리"의 핵심)

## 7. 사용법

```bash
# 0) 클론 + lfs
git clone https://github.com/supertone-inc/supertonic.git
cd supertonic
git lfs install
git clone https://huggingface.co/Supertone/supertonic-2 assets

# 1) 의존성
pip install torch onnx onnxruntime numpy soundfile

# 2) 각 모듈 단독 검증
python analysis/torch_vocoder.py
python analysis/torch_duration_predictor.py
python analysis/torch_text_encoder.py
python analysis/torch_vector_estimator.py

# 3) 전체 파이프라인 비교
python analysis/verify_end_to_end.py
```

## 8. 참고

- 논문: arXiv [2503.23108](https://arxiv.org/abs/2503.23108) (main), [2509.11084](https://arxiv.org/abs/2509.11084) (LARoPE), [2509.19091](https://arxiv.org/abs/2509.19091) (SPFM)
- 모델: [huggingface.co/Supertone/supertonic-2](https://huggingface.co/Supertone/supertonic-2)
- 라이선스: 코드 MIT / 모델 OpenRAIL-M — 재배포 시 확인 필요
