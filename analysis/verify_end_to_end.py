"""End-to-end PyTorch vs ONNX pipeline verification.

Runs the full TTS pipeline (duration_predictor -> text_encoder -> vector_estimator×N -> vocoder)
in both ONNX (reference) and our reconstructed PyTorch modules, using real voice styles and
multiple test texts, and reports per-stage diffs + final waveform diff.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import torch

# ensure py/ is importable for helper.py's UnicodeProcessor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "py"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helper import UnicodeProcessor, get_latent_mask, length_to_mask  # type: ignore
from torch_vocoder import Vocoder, load_vocoder_weights
from torch_duration_predictor import DurationPredictor, load_dp_weights
from torch_text_encoder import TextEncoder, load_text_encoder_weights
from torch_vector_estimator import VectorField, load_ve_weights

ONNX_DIR = os.path.join(ROOT, "assets", "onnx")
STYLE_DIR = os.path.join(ROOT, "assets", "voice_styles")


def load_style_json(path: str):
    with open(path, "r") as f:
        s = json.load(f)
    ttl = np.array(s["style_ttl"]["data"], dtype=np.float32).flatten().reshape(s["style_ttl"]["dims"][1:])
    dp  = np.array(s["style_dp"]["data"],  dtype=np.float32).flatten().reshape(s["style_dp"]["dims"][1:])
    return ttl[None], dp[None]   # [1, 50, 256], [1, 8, 16]


def load_torch_models():
    voc = Vocoder();            load_vocoder_weights(voc,     os.path.join(ONNX_DIR, "vocoder.onnx"))
    dp  = DurationPredictor();  load_dp_weights(dp,           os.path.join(ONNX_DIR, "duration_predictor.onnx"))
    te  = TextEncoder();        load_text_encoder_weights(te, os.path.join(ONNX_DIR, "text_encoder.onnx"))
    ve  = VectorField();        load_ve_weights(ve,           os.path.join(ONNX_DIR, "vector_estimator.onnx"))
    for m in (voc, dp, te, ve): m.eval()
    return voc, dp, te, ve


def load_onnx_sessions():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    return {
        "dp":  ort.InferenceSession(os.path.join(ONNX_DIR, "duration_predictor.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]),
        "te":  ort.InferenceSession(os.path.join(ONNX_DIR, "text_encoder.onnx"),      sess_options=opts, providers=["CPUExecutionProvider"]),
        "ve":  ort.InferenceSession(os.path.join(ONNX_DIR, "vector_estimator.onnx"),  sess_options=opts, providers=["CPUExecutionProvider"]),
        "voc": ort.InferenceSession(os.path.join(ONNX_DIR, "vocoder.onnx"),           sess_options=opts, providers=["CPUExecutionProvider"]),
    }


def sample_noisy_latent(duration_sec: np.ndarray, sample_rate=44100, base_chunk=512, kc=6, ldim=24, seed=0):
    rng = np.random.RandomState(seed)
    bsz = len(duration_sec)
    wav_len_max = duration_sec.max() * sample_rate
    wav_lengths = (duration_sec * sample_rate).astype(np.int64)
    chunk = base_chunk * kc
    latent_len = int((wav_len_max + chunk - 1) // chunk)
    noisy = rng.randn(bsz, ldim * kc, latent_len).astype(np.float32)
    lat_mask = get_latent_mask(wav_lengths, base_chunk, kc).astype(np.float32)
    return noisy * lat_mask, lat_mask


def diff_stats(a, b, label):
    d = np.abs(a - b)
    return {"label": label, "shape": list(a.shape), "max": float(d.max()), "mean": float(d.mean()), "onnx_range": (float(a.min()), float(a.max()))}


def run_pipeline(text: str, lang: str, style_ttl: np.ndarray, style_dp: np.ndarray,
                 torch_models, onnx_sessions, text_processor: UnicodeProcessor,
                 total_step: int = 5, speed: float = 1.0):
    """Run full pipeline in both PyTorch and ONNX, tap at each stage, return diffs."""
    voc_t, dp_t, te_t, ve_t = torch_models
    dp_s, te_s, ve_s, voc_s = onnx_sessions["dp"], onnx_sessions["te"], onnx_sessions["ve"], onnx_sessions["voc"]

    # 1) Text preprocessing
    text_ids, text_mask = text_processor([text], [lang])
    text_mask = text_mask.astype(np.float32)

    diffs = []

    # 2) Duration predictor
    dur_onnx = dp_s.run(None, {"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask})[0]
    with torch.no_grad():
        dur_torch = dp_t(torch.from_numpy(text_ids), torch.from_numpy(style_dp), torch.from_numpy(text_mask)).numpy()
    diffs.append(diff_stats(dur_onnx, dur_torch, "duration"))
    dur = dur_onnx / speed  # use ONNX duration for downstream to isolate errors

    # 3) Text encoder
    te_onnx = te_s.run(None, {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask})[0]
    with torch.no_grad():
        te_torch = te_t(torch.from_numpy(text_ids), torch.from_numpy(style_ttl), torch.from_numpy(text_mask)).numpy()
    diffs.append(diff_stats(te_onnx, te_torch, "text_emb"))

    # 4) Flow matching: identical z_0 seed, run total_step iterations
    noisy_init, lat_mask = sample_noisy_latent(dur, seed=0)
    total_np = np.array([total_step] * len(dur), dtype=np.float32)
    xt_onnx = noisy_init.copy()
    xt_torch = torch.from_numpy(noisy_init.copy())
    for step in range(total_step):
        cur = np.array([step] * len(dur), dtype=np.float32)
        xt_onnx = ve_s.run(None, {
            "noisy_latent": xt_onnx, "text_emb": te_onnx, "style_ttl": style_ttl,
            "text_mask": text_mask, "latent_mask": lat_mask,
            "current_step": cur, "total_step": total_np,
        })[0]
        with torch.no_grad():
            xt_torch = ve_t(
                xt_torch, torch.from_numpy(te_torch), torch.from_numpy(style_ttl),
                torch.from_numpy(lat_mask), torch.from_numpy(text_mask),
                torch.from_numpy(cur), torch.from_numpy(total_np),
            )
    diffs.append(diff_stats(xt_onnx, xt_torch.numpy(), f"latent_after_{total_step}steps"))

    # 5) Vocoder
    wav_onnx = voc_s.run(None, {"latent": xt_onnx})[0]
    with torch.no_grad():
        wav_torch = voc_t(xt_torch).numpy()
    # Trim to the valid duration
    sr = 44100
    n_samples = int(sr * dur[0])
    w1 = wav_onnx[0, :n_samples]
    w2 = wav_torch[0, :n_samples]
    diffs.append(diff_stats(w1, w2, "wav"))
    return diffs, w1, w2


def main():
    print(f"Loading models...", flush=True)
    t0 = time.time()
    torch_models = load_torch_models()
    onnx_sessions = load_onnx_sessions()
    text_processor = UnicodeProcessor(os.path.join(ONNX_DIR, "unicode_indexer.json"))
    print(f"  loaded in {time.time()-t0:.1f}s\n")

    test_cases = [
        ("en", "M1", "Hello, this is a short sentence."),
        ("en", "F3", "This morning, I took a walk in the park, and the sound of the birds was pleasant."),
        ("ko", "M4", "안녕하세요. 음성 합성 테스트입니다."),
        ("es", "F1", "Hola, esto es una prueba de sintesis."),
    ]

    for lang, speaker, text in test_cases:
        ttl, dp = load_style_json(os.path.join(STYLE_DIR, f"{speaker}.json"))
        print(f"=== [{lang}] {speaker}: {text[:50]}...")
        diffs, _, _ = run_pipeline(text, lang, ttl, dp, torch_models, onnx_sessions, text_processor, total_step=5)
        for d in diffs:
            print(f"  {d['label']:30s} shape={str(d['shape']):24s} max|Δ|={d['max']:.4e} mean={d['mean']:.4e} onnx_range=[{d['onnx_range'][0]:.3f}, {d['onnx_range'][1]:.3f}]")
        print()


if __name__ == "__main__":
    main()
