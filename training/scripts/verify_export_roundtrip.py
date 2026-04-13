"""Roundtrip test for export_onnx.py (no training required).

Builds fake "trained checkpoints" from the released ONNX weights, then runs the
export pipeline, then runs the exported ONNX back-to-back with the original and
compares outputs. If diff is small, the export contract is correct.

Caveat: the AE encoder portion isn't in the shipped ONNX (only decoder is), so the
`ae` fake-ckpt only has decoder weights. Vocoder export only needs the decoder +
latent_mean/std, which is enough for this test. Other exports (DP, text_encoder,
vector_estimator) load all their weights from shipped ONNX.
"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from training.models.ae_decoder import AEDecoder
# Verified ONNX loaders from analysis/
from torch_vocoder            import load_vocoder_weights             # type: ignore
from torch_duration_predictor import DurationPredictor, load_dp_weights       # type: ignore
from torch_text_encoder       import TextEncoder, load_text_encoder_weights   # type: ignore
from torch_vector_estimator   import VectorField, load_ve_weights             # type: ignore
from training.scripts.export_onnx import (
    export_vocoder, export_duration_predictor, export_text_encoder, export_vector_estimator,
    export_vector_estimator_cfg,
)
from training.models.vector_field_cfg import VectorFieldCFG


ASSETS = ROOT / "assets" / "onnx"
TOL = 1e-4


def _load_ae_decoder_from_onnx() -> tuple[AEDecoder, torch.Tensor, torch.Tensor]:
    """Load decoder weights from vocoder.onnx into our AEDecoder; return decoder + mean + std."""
    dec = AEDecoder()
    # Manual mapping (same as training/scripts/verify_decoder_weights.py)
    inits = {i.name: numpy_helper.to_array(i)
             for i in onnx.load(str(ASSETS / "vocoder.onnx")).graph.initializer}

    def cp(p, a):
        with torch.no_grad():
            p.copy_(torch.from_numpy(np.ascontiguousarray(a).reshape(p.shape).astype("float32")))

    cp(dec.stem.weight, inits["onnx::Conv_1440"])
    cp(dec.stem.bias,   inits["onnx::Conv_1441"])
    for i, blk in enumerate(dec.convnext):
        p = f"tts.ae.decoder.convnext.{i}"
        cp(blk.dwconv.weight,   inits[f"{p}.dwconv.net.weight"])
        cp(blk.dwconv.bias,     inits[f"{p}.dwconv.net.bias"])
        cp(blk.norm.weight,     inits[f"{p}.norm.norm.weight"])
        cp(blk.norm.bias,       inits[f"{p}.norm.norm.bias"])
        cp(blk.pwconv1.weight,  inits[f"{p}.pwconv1.weight"])
        cp(blk.pwconv1.bias,    inits[f"{p}.pwconv1.bias"])
        cp(blk.pwconv2.weight,  inits[f"{p}.pwconv2.weight"])
        cp(blk.pwconv2.bias,    inits[f"{p}.pwconv2.bias"])
        cp(blk.gamma,           inits[f"{p}.gamma"])
    cp(dec.final_norm.weight,       inits["tts.ae.decoder.final_norm.norm.weight"])
    cp(dec.final_norm.bias,         inits["tts.ae.decoder.final_norm.norm.bias"])
    cp(dec.final_norm.running_mean, inits["tts.ae.decoder.final_norm.norm.running_mean"])
    cp(dec.final_norm.running_var,  inits["tts.ae.decoder.final_norm.norm.running_var"])
    cp(dec.head_layer1.weight, inits["tts.ae.decoder.head.layer1.net.weight"])
    cp(dec.head_layer1.bias,   inits["tts.ae.decoder.head.layer1.net.bias"])
    cp(dec.head_act.weight,    inits["onnx::PRelu_1505"].reshape(-1))
    cp(dec.head_layer2.weight, inits["tts.ae.decoder.head.layer2.weight"])
    dec.eval()

    mean = torch.from_numpy(inits["tts.ae.latent_mean"].astype("float32")).flatten()
    std  = torch.from_numpy(inits["tts.ae.latent_std"].astype("float32")).flatten()
    return dec, mean, std


def _make_fake_ckpts(tmp: Path):
    """Save fake 'trained ckpts' built from released ONNX weights."""
    # AE ckpt: only decoder + stats (encoder not available in ONNX)
    dec, mean, std = _load_ae_decoder_from_onnx()
    ae_sd = {f"decoder.{k}": v for k, v in dec.state_dict().items()}
    torch.save({"ae": ae_sd}, tmp / "ae.pt")
    torch.save({"mean": mean, "std": std, "n_frames": 1, "n_utterances": 1}, tmp / "stats.pt")

    # DP ckpt
    dp = DurationPredictor(); load_dp_weights(dp, str(ASSETS / "duration_predictor.onnx")); dp.eval()
    torch.save({"dp": dp.state_dict()}, tmp / "dp.pt")

    # TTL ckpt (text_encoder + vector_field + uncond_masker placeholder for CFG export)
    te = TextEncoder(); load_text_encoder_weights(te, str(ASSETS / "text_encoder.onnx")); te.eval()
    vf = VectorField(); load_ve_weights(vf, str(ASSETS / "vector_estimator.onnx")); vf.eval()
    # Synthesize a fake uncond_masker state (shapes must match UncondMasker(defaults))
    fake_um = {
        "uncond_text":  torch.zeros(1, 256, 1),
        "uncond_style": torch.zeros(1, 50, 256),
    }
    torch.save({
        "text_encoder":  te.state_dict(),
        "vector_field":  vf.state_dict(),
        "uncond_masker": fake_um,
    }, tmp / "ttl.pt")
    return dec, mean, std


def _diff(a, b, label):
    a = np.asarray(a); b = np.asarray(b)
    d = np.abs(a - b)
    print(f"  {label:18s} max|Δ|={d.max():.4e}  mean={d.mean():.4e}  shape={a.shape}")
    return d.max()


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print(f"[setup] building fake ckpts in {tmp}")
        _make_fake_ckpts(tmp)

        out_dir = tmp / "exported"; out_dir.mkdir()
        device = torch.device("cpu")

        print("\n[export]")
        with torch.no_grad():
            export_vocoder(tmp / "ae.pt", tmp / "stats.pt", out_dir / "vocoder.onnx",
                           device, opset=19)
            export_duration_predictor(tmp / "dp.pt", out_dir / "duration_predictor.onnx",
                                      device, opset=19)
            export_text_encoder(tmp / "ttl.pt", out_dir / "text_encoder.onnx",
                                device, opset=19)
            export_vector_estimator(tmp / "ttl.pt", out_dir / "vector_estimator.onnx",
                                    device, opset=19)
            export_vector_estimator_cfg(tmp / "ttl.pt", out_dir / "vector_estimator_cfg.onnx",
                                        device, opset=19)

        print("\n[roundtrip compare]")
        np.random.seed(0)

        # --- vocoder ---
        ref = ort.InferenceSession(str(ASSETS / "vocoder.onnx"), providers=["CPUExecutionProvider"])
        new = ort.InferenceSession(str(out_dir / "vocoder.onnx"), providers=["CPUExecutionProvider"])
        latent = np.random.randn(2, 144, 17).astype("float32") * 0.3
        yr = ref.run(None, {"latent": latent})[0]
        yn = new.run(None, {"latent": latent})[0]
        d_v = _diff(yr, yn, "vocoder")

        # --- DP ---
        ref = ort.InferenceSession(str(ASSETS / "duration_predictor.onnx"), providers=["CPUExecutionProvider"])
        new = ort.InferenceSession(str(out_dir / "duration_predictor.onnx"), providers=["CPUExecutionProvider"])
        text_ids  = np.random.randint(0, 162, size=(2, 30)).astype("int64")
        style_dp  = np.random.randn(2, 8, 16).astype("float32") * 0.2
        text_mask = np.ones((2, 1, 30), dtype="float32"); text_mask[0, 0, 25:] = 0
        yr = ref.run(None, {"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask})[0]
        yn = new.run(None, {"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask})[0]
        d_d = _diff(yr, yn, "duration_predictor")

        # --- text_encoder ---
        ref = ort.InferenceSession(str(ASSETS / "text_encoder.onnx"), providers=["CPUExecutionProvider"])
        new = ort.InferenceSession(str(out_dir / "text_encoder.onnx"), providers=["CPUExecutionProvider"])
        style_ttl = np.random.randn(2, 50, 256).astype("float32") * 0.3
        yr = ref.run(None, {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask})[0]
        yn = new.run(None, {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask})[0]
        d_t = _diff(yr, yn, "text_encoder")

        # --- vector_estimator ---
        ref = ort.InferenceSession(str(ASSETS / "vector_estimator.onnx"), providers=["CPUExecutionProvider"])
        new = ort.InferenceSession(str(out_dir / "vector_estimator.onnx"), providers=["CPUExecutionProvider"])
        B, L, T = 2, 17, 25
        noisy  = np.random.randn(B, 144, L).astype("float32") * 0.3
        text   = np.random.randn(B, 256, T).astype("float32") * 0.3
        sttl   = np.random.randn(B, 50, 256).astype("float32") * 0.3
        lm     = np.ones((B, 1, L), dtype="float32"); lm[0,0,15:] = 0
        tm     = np.ones((B, 1, T), dtype="float32"); tm[0,0,22:] = 0
        cs     = np.array([1.0, 2.0], dtype="float32")
        ts     = np.array([5.0, 5.0], dtype="float32")
        inp = dict(noisy_latent=noisy, text_emb=text, style_ttl=sttl,
                   latent_mask=lm, text_mask=tm, current_step=cs, total_step=ts)
        yr = ref.run(None, inp)[0]
        yn = new.run(None, inp)[0]
        d_ve = _diff(yr, yn, "vector_estimator")

        # --- vector_estimator_cfg: check cfg_scale=0 matches non-CFG exactly ---
        cfg_sess = ort.InferenceSession(str(out_dir / "vector_estimator_cfg.onnx"),
                                        providers=["CPUExecutionProvider"])
        inp_cfg = dict(inp); inp_cfg["cfg_scale"] = np.zeros(B, dtype="float32")
        yc = cfg_sess.run(None, inp_cfg)[0]
        d_vecfg = _diff(yr, yc, "vector_estimator_cfg (cfg=0)")

    print("\n[result]")
    fails = []
    for lbl, v in [("vocoder", d_v), ("duration_predictor", d_d),
                   ("text_encoder", d_t), ("vector_estimator", d_ve),
                   ("vector_estimator_cfg@0", d_vecfg)]:
        ok = v < TOL
        print(f"  {lbl:18s} {'OK' if ok else 'FAIL'}  (max|Δ|={v:.4e}, tol={TOL:.0e})")
        if not ok: fails.append(lbl)
    if fails:
        print(f"\n[FAIL] {len(fails)} module(s) failed roundtrip: {fails}")
        sys.exit(1)
    print("\n[OK] All exports roundtrip-verified.")


if __name__ == "__main__":
    main()
