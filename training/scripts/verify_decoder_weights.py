"""Verify our pure AEDecoder can load weights from vocoder.onnx (minus TTL wrappers).

This confirms:
  1. Our AEDecoder has the exact same structure as the shipped AE decoder.
  2. We can optionally warm-start AE training from the pretrained decoder weights.

Compare the decoder-only portion of vocoder.onnx output against our AEDecoder output,
given the same latent (post un-chunk & de-normalize, i.e. in native AE latent space).
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from training.models.ae_decoder import AEDecoder

ONNX_PATH = os.path.join(ROOT, "assets", "onnx", "vocoder.onnx")


def load_decoder_from_onnx(decoder: AEDecoder, onnx_path: str):
    """Load just the decoder portion of vocoder.onnx into our AEDecoder (skip un-chunk/denorm wrappers)."""
    inits = {i.name: numpy_helper.to_array(i) for i in onnx.load(onnx_path).graph.initializer}

    def cp(p, a, name):
        a = a if a.shape == tuple(p.shape) else a.reshape(tuple(p.shape))
        with torch.no_grad():
            p.copy_(torch.from_numpy(a.astype("float32")))

    # Stem
    cp(decoder.stem.weight, inits["onnx::Conv_1440"], "stem.w")
    cp(decoder.stem.bias,   inits["onnx::Conv_1441"], "stem.b")

    # 10× ConvNeXt
    for i, blk in enumerate(decoder.convnext):
        p = f"tts.ae.decoder.convnext.{i}"
        cp(blk.dwconv.weight, inits[f"{p}.dwconv.net.weight"], "dw.w")
        cp(blk.dwconv.bias,   inits[f"{p}.dwconv.net.bias"],   "dw.b")
        cp(blk.norm.weight,   inits[f"{p}.norm.norm.weight"],  "ln.w")
        cp(blk.norm.bias,     inits[f"{p}.norm.norm.bias"],    "ln.b")
        cp(blk.pwconv1.weight, inits[f"{p}.pwconv1.weight"], "pw1.w")
        cp(blk.pwconv1.bias,   inits[f"{p}.pwconv1.bias"],   "pw1.b")
        cp(blk.pwconv2.weight, inits[f"{p}.pwconv2.weight"], "pw2.w")
        cp(blk.pwconv2.bias,   inits[f"{p}.pwconv2.bias"],   "pw2.b")
        cp(blk.gamma,          inits[f"{p}.gamma"],          "gamma")

    # final BatchNorm
    cp(decoder.final_norm.weight,       inits["tts.ae.decoder.final_norm.norm.weight"],       "fn.w")
    cp(decoder.final_norm.bias,         inits["tts.ae.decoder.final_norm.norm.bias"],         "fn.b")
    cp(decoder.final_norm.running_mean, inits["tts.ae.decoder.final_norm.norm.running_mean"], "fn.rm")
    cp(decoder.final_norm.running_var,  inits["tts.ae.decoder.final_norm.norm.running_var"],  "fn.rv")

    # Head
    cp(decoder.head_layer1.weight, inits["tts.ae.decoder.head.layer1.net.weight"], "h1.w")
    cp(decoder.head_layer1.bias,   inits["tts.ae.decoder.head.layer1.net.bias"],   "h1.b")
    cp(decoder.head_act.weight,    inits["onnx::PRelu_1505"].reshape(-1),          "hp.w")
    cp(decoder.head_layer2.weight, inits["tts.ae.decoder.head.layer2.weight"],     "h2.w")


def main():
    # Our AEDecoder expects native AE latent (no TTL wrapping).
    # vocoder.onnx takes TTL latent [B, 144, L_ttl] and internally:
    #   (1) /= normalizer_scale (0.25)
    #   (2) un-chunk 6× → [B, 24, L_ae=6·L_ttl]
    #   (3) * latent_std + latent_mean
    # To compare: we feed vocoder.onnx a random TTL latent and extract the
    # value post-(3) (i.e. native AE latent), then feed that same tensor to our AEDecoder.

    from onnx import helper, TensorProto
    m = onnx.load(ONNX_PATH)
    # Tap the native AE latent (after un-chunk + de-norm)
    tap_name = "/Add_output_0"   # the "Add" that applies latent_mean (see analysis/torch_vocoder.py)
    m.graph.output.append(helper.make_tensor_value_info(tap_name, TensorProto.FLOAT, None))
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vocoder_ae_tap.onnx")
    onnx.save(m, tmp_path)

    np.random.seed(0)
    B, L_ttl = 2, 17
    latent_ttl = np.random.randn(B, 144, L_ttl).astype(np.float32) * 0.3
    sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    outs = dict(zip(out_names, sess.run(None, {"latent": latent_ttl})))
    ae_latent = outs[tap_name]                       # [B, 24, L_ae=6*L_ttl]
    wav_onnx = outs["wav_tts"]                        # [B, L_ae*512]
    print(f"AE latent shape from tap: {ae_latent.shape}")
    print(f"wav onnx shape:           {wav_onnx.shape}")

    # Build our decoder + load weights
    decoder = AEDecoder()
    load_decoder_from_onnx(decoder, ONNX_PATH)
    decoder.eval()

    with torch.no_grad():
        wav_torch = decoder(torch.from_numpy(ae_latent)).numpy()

    diff = np.abs(wav_onnx - wav_torch)
    print(f"wav max|Δ|: {diff.max():.4e}   mean|Δ|: {diff.mean():.4e}")
    print(f"wav ranges: onnx=[{wav_onnx.min():.3f}, {wav_onnx.max():.3f}]  torch=[{wav_torch.min():.3f}, {wav_torch.max():.3f}]")
    if diff.max() < 1e-4:
        print("OK: pure AEDecoder = shipped AE decoder (structural mirror of encoder).")
    else:
        print("WARNING: diff > 1e-4, check implementation.")

    # clean up tap file
    os.remove(tmp_path)


if __name__ == "__main__":
    main()
