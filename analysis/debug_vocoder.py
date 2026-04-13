"""Expose intermediate outputs from vocoder.onnx and compare against torch stages."""
import os, sys, numpy as np, torch, onnx, onnxruntime as ort
from onnx import numpy_helper, helper, TensorProto

sys.path.insert(0, os.path.dirname(__file__))
from torch_vocoder import Vocoder, load_vocoder_weights, causal_pad

ONNX = os.path.join(os.path.dirname(__file__), "..", "assets", "onnx", "vocoder.onnx")
TAPS = [
    ("after_div",       "/Div_output_0"),
    ("after_reshape4d", "/Reshape_output_0"),
    ("after_transpose", "/Transpose_output_0"),
    ("after_reshape3d", "/Reshape_1_output_0"),
    ("after_mul_std",   "/Mul_output_0"),
    ("after_add_mean",  "/Add_output_0"),
    ("after_stem",      "/decoder/embed/net/Conv_output_0"),
    ("after_block0",    "/decoder/convnext.0/Add_output_0"),
    ("after_block9",    "/decoder/convnext.9/Add_output_0"),
    ("after_bn",        "/decoder/final_norm/BatchNormalization_output_0"),
    ("after_head1",     "/decoder/head/layer1/net/Conv_output_0"),
    ("after_prelu",     "/decoder/head/act/PRelu_output_0"),
    ("after_head2",     "/decoder/head/layer2/Conv_output_0"),
    ("final",           "wav_tts"),
]

def augment_and_run(x_np):
    m = onnx.load(ONNX)
    for label, out_name in TAPS:
        vi = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, None)
        m.graph.output.append(vi)
    tmp = os.path.join(os.path.dirname(__file__), "_vocoder_tap.onnx")
    onnx.save(m, tmp)
    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    # Outputs include original + appended. Run by name.
    out_names = [o.name for o in sess.get_outputs()]
    results = sess.run(out_names, {"latent": x_np})
    named = dict(zip(out_names, results))
    return named


def torch_stages(model: Vocoder, latent: torch.Tensor):
    """Compute stage-by-stage in torch matching TAPS."""
    stages = {}
    B = latent.shape[0]
    L = latent.shape[-1]

    x = latent / model.normalizer_scale
    stages["after_div"] = x

    x = x.reshape(B, model.ldim, model.kc, L)
    stages["after_reshape4d"] = x

    x = x.permute(0, 1, 3, 2).contiguous()
    stages["after_transpose"] = x

    x = x.reshape(B, model.ldim, L * model.kc)
    stages["after_reshape3d"] = x

    x = x * model.latent_std
    stages["after_mul_std"] = x

    x = x + model.latent_mean
    stages["after_add_mean"] = x

    x = causal_pad(x, model.stem_ksz, 1)
    x = model.stem(x)
    stages["after_stem"] = x

    for i, blk in enumerate(model.convnext):
        x = blk(x)
        if i == 0: stages["after_block0"] = x
        if i == 9: stages["after_block9"] = x

    x = model.final_norm(x)
    stages["after_bn"] = x

    x = causal_pad(x, model.head_ksz, 1)
    x = model.head_layer1(x)
    stages["after_head1"] = x

    x = model.head_act(x)
    stages["after_prelu"] = x

    x = model.head_layer2(x)
    stages["after_head2"] = x

    wav = x.transpose(1, 2).reshape(B, -1)
    stages["final"] = wav
    return stages


def main():
    np.random.seed(0); torch.manual_seed(0)
    B, L = 2, 17
    x_np = np.random.randn(B, 144, L).astype(np.float32) * 0.3

    model = Vocoder()
    load_vocoder_weights(model, ONNX)
    model.eval()

    with torch.no_grad():
        stages_t = torch_stages(model, torch.from_numpy(x_np))

    outs = augment_and_run(x_np)

    print(f"{'stage':20s} {'onnx_shape':25s} {'torch_shape':25s} {'max_abs_diff':>14s} {'mean':>12s}")
    print("-" * 110)
    for label, out_name in TAPS:
        yo = outs[out_name]
        yt = stages_t[label].cpu().numpy()
        if yo.shape != yt.shape:
            print(f"{label:20s} {str(yo.shape):25s} {str(yt.shape):25s}   SHAPE MISMATCH")
            continue
        d = np.abs(yo - yt)
        print(f"{label:20s} {str(yo.shape):25s} {str(yt.shape):25s}   {d.max():.4e}   {d.mean():.4e}")


if __name__ == "__main__":
    main()
