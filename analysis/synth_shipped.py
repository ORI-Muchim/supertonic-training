"""Synthesize Korean text via the FULL SHIPPED Supertonic pipeline.

This is the demo-quality reference: shipped TextEncoder + VectorEstimator + DP +
Vocoder + a shipped voice_styles JSON. If THIS sounds clean on KSS-style input,
the architecture isn't the ceiling — our from-scratch single-speaker training
data is the bottleneck.

Usage:
    python analysis/synth_shipped.py
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
import numpy as np
import soundfile as sf
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(ROOT))

from helper import UnicodeProcessor, Style, TextToSpeech   # type: ignore


def load_voice_style(path: str) -> Style:
    with open(path, "r") as f:
        d = json.load(f)
    ttl = np.array(d["style_ttl"]["data"], dtype=np.float32).reshape(d["style_ttl"]["dims"])
    dp = np.array(d["style_dp"]["data"], dtype=np.float32).reshape(d["style_dp"]["dims"])
    return Style(ttl, dp)


def main():
    onnx_dir = ROOT / "assets" / "onnx"
    cfgs_path = onnx_dir / "tts.json"
    with open(cfgs_path) as f:
        cfgs = json.load(f)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    text_proc = UnicodeProcessor(str(onnx_dir / "unicode_indexer.json"))
    dp_ort   = ort.InferenceSession(str(onnx_dir / "duration_predictor.onnx"), providers=providers)
    te_ort   = ort.InferenceSession(str(onnx_dir / "text_encoder.onnx"),       providers=providers)
    ve_ort   = ort.InferenceSession(str(onnx_dir / "vector_estimator.onnx"),   providers=providers)
    voc_ort  = ort.InferenceSession(str(onnx_dir / "vocoder.onnx"),            providers=providers)
    tts = TextToSpeech(cfgs, text_proc, dp_ort, te_ort, ve_ort, voc_ort)

    sample_rate = cfgs["ae"]["sample_rate"]

    # Same test sentences we synthesized with our paper-faithful pipeline
    tests = [
        ("shipped_01_greeting",  "안녕하세요 반갑습니다"),
        ("shipped_02_weather",   "오늘 날씨가 정말 좋네요"),
        ("shipped_03_narrative", "어제 저녁에 친구를 만나서 영화를 보러 갔어요"),
        ("shipped_04_tech",      "인공지능 음성합성 기술이 빠르게 발전하고 있습니다"),
        ("shipped_05_casual",    "커피 한 잔 마시면서 책을 읽고 싶어요"),
        ("shipped_06_short",     "감사합니다"),
        ("shipped_07_refsame",   "그는 괜찮은 척하려고 애쓰는 것 같았다"),
    ]

    # Use F1 (Korean female) as reference voice
    style_path = ROOT / "assets" / "voice_styles" / "F1.json"
    style = load_voice_style(str(style_path))
    print(f"[info] using voice style: {style_path.name}  style_ttl={style.ttl.shape}  style_dp={style.dp.shape}")

    for tag, text in tests:
        wav, dur = tts(text, lang="ko", style=style, total_step=5, speed=1.05)
        wav_np = wav.squeeze(0).astype(np.float32)
        dur_f = float(np.asarray(dur).reshape(-1)[0])
        peak = float(np.abs(wav_np).max())
        rms = float(np.sqrt((wav_np**2).mean()))
        clip = float((np.abs(wav_np) >= 1.0 - 1e-4).mean() * 100)
        out_path = ROOT / f"out_{tag}.wav"
        sf.write(str(out_path), wav_np, sample_rate)
        print(f"[{tag}] dur={dur_f:.2f}s  peak={peak:.3f}  rms={rms:.3f}  clip={clip:.2f}%  -> {out_path.name}")


if __name__ == "__main__":
    main()
