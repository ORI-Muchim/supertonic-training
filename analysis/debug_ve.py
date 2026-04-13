"""Stage debug for vector_estimator."""
import os, sys, numpy as np, torch, onnx, onnxruntime as ort
from onnx import helper, TensorProto
sys.path.insert(0, os.path.dirname(__file__))
from torch_vector_estimator import VectorField, load_ve_weights

ONNX = os.path.join(os.path.dirname(__file__), "..", "assets", "onnx", "vector_estimator.onnx")

# Taps for block 0
TAPS = [
    ("blk3_end",           "/vector_field/main_blocks.23/Mul_1_output_0"),
    ("lastcn_0",           "/vector_field/last_convnext/convnext.0/Mul_3_output_0"),
    ("lastcn_3",           "/vector_field/last_convnext/convnext.3/Mul_3_output_0"),
    ("final",              "denoised_latent"),
]


def augment(names):
    m = onnx.load(ONNX)
    for nm in names:
        m.graph.output.append(helper.make_tensor_value_info(nm, TensorProto.FLOAT, None))
    tmp = os.path.join(os.path.dirname(__file__), "_ve_tap.onnx")
    onnx.save(m, tmp)
    return tmp


def torch_stages(model, inputs):
    stages = {}
    t = inputs["current_step"] / inputs["total_step"]
    time_emb = model.time_encoder(t)
    x = model.proj_in(inputs["noisy_latent"]) * inputs["latent_mask"]
    for blk in model.main_blocks:
        x = blk(x, time_emb, inputs["text_emb"], inputs["style_ttl"],
                inputs["latent_mask"], inputs["text_mask"], model.style_prototype)
    stages["blk3_end"] = x
    for i, lyr in enumerate(model.last_convnext.layers):
        x = lyr(x, inputs["latent_mask"])
        if i == 0: stages["lastcn_0"] = x
        if i == 3: stages["lastcn_3"] = x
    x = model.proj_out(x) * inputs["latent_mask"]
    stages["final"] = x
    return stages


def main():
    np.random.seed(0); torch.manual_seed(0)
    B, L, T = 2, 17, 25
    inputs_np = {
        "noisy_latent": np.random.randn(B, 144, L).astype(np.float32) * 0.3,
        "text_emb":     np.random.randn(B, 256, T).astype(np.float32) * 0.3,
        "style_ttl":    np.random.randn(B, 50, 256).astype(np.float32) * 0.3,
        "latent_mask":  np.zeros((B, 1, L), dtype=np.float32),
        "text_mask":    np.zeros((B, 1, T), dtype=np.float32),
        "current_step": np.array([1.0, 2.0], dtype=np.float32),
        "total_step":   np.array([5.0, 5.0], dtype=np.float32),
    }
    inputs_np["latent_mask"][0,0,:15]=1; inputs_np["latent_mask"][1,0,:]=1
    inputs_np["text_mask"][0,0,:22]=1; inputs_np["text_mask"][1,0,:]=1

    tap_onnx = augment([t[1] for t in TAPS])
    sess = ort.InferenceSession(tap_onnx, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    outs = dict(zip(names, sess.run(None, inputs_np)))

    model = VectorField()
    load_ve_weights(model, ONNX)
    model.eval()
    inputs_t = {k: torch.from_numpy(v) for k, v in inputs_np.items()}
    with torch.no_grad():
        stg = torch_stages(model, inputs_t)

    print(f"{'stage':20s} {'onnx_shape':25s} {'torch_shape':25s} {'max|Δ|':>14s} {'mean':>12s}")
    print("-" * 95)
    for lbl, nm in TAPS:
        yo = outs[nm]
        yt = stg[lbl].cpu().numpy()
        if yo.shape != yt.shape:
            print(f"{lbl:20s} {str(yo.shape):25s} {str(yt.shape):25s}   SHAPE MISMATCH")
            continue
        d = np.abs(yo - yt)
        print(f"{lbl:20s} {str(yo.shape):25s} {str(yt.shape):25s}   {d.max():.4e}   {d.mean():.4e}")


if __name__ == "__main__":
    main()
