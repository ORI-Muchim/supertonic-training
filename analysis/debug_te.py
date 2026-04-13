"""Stage debug for text_encoder."""
import os, sys, numpy as np, torch, onnx, onnxruntime as ort
from onnx import helper, TensorProto
sys.path.insert(0, os.path.dirname(__file__))
from torch_text_encoder import TextEncoder, load_text_encoder_weights

ONNX = os.path.join(os.path.dirname(__file__), "..", "assets", "onnx", "text_encoder.onnx")

TAPS = [
    ("after_main",     "/text_encoder/Add_output_0"),  # main text encoder final add (conv skip around attn)
    ("after_attn1",    "/speech_prompted_text_encoder/Add_output_0"),  # after attn1 residual
    ("after_attn2",    "/speech_prompted_text_encoder/Add_1_output_0"),  # after attn2 residual
    ("final",          "text_emb"),
]


def dump_extra(m, names):
    for nm in names:
        m.graph.output.append(helper.make_tensor_value_info(nm, TensorProto.FLOAT, None))


def main():
    np.random.seed(0); torch.manual_seed(0)
    B, T = 2, 35
    text_ids = np.random.randint(0, 162, size=(B, T)).astype(np.int64)
    style_ttl = np.random.randn(B, 50, 256).astype(np.float32) * 0.3
    text_mask = np.zeros((B, 1, T), dtype=np.float32); text_mask[0,0,:30]=1; text_mask[1,0,:]=1

    m = onnx.load(ONNX)
    dump_extra(m, [t[1] for t in TAPS if t[1] != "text_emb"])
    tmp = os.path.join(os.path.dirname(__file__), "_te_tap.onnx")
    onnx.save(m, tmp)
    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    outs = dict(zip(names, sess.run(None, {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask})))

    model = TextEncoder(); load_text_encoder_weights(model, ONNX); model.eval()
    tids = torch.from_numpy(text_ids); sty = torch.from_numpy(style_ttl); tm = torch.from_numpy(text_mask)
    with torch.no_grad():
        x_main = model.text_encoder(tids, tm)
        x_after_attn1 = x_main.transpose(1, 2) + model.speech_prompted_text_encoder.attention1(x_main.transpose(1,2), sty, tm)
        x_after_attn2 = x_after_attn1 + model.speech_prompted_text_encoder.attention2(x_after_attn1, sty, tm)
        x_ln = model.speech_prompted_text_encoder.norm(x_after_attn2).transpose(1, 2) * tm

    tstages = {
        "after_main":  x_main,
        "after_attn1": x_after_attn1,
        "after_attn2": x_after_attn2,
        "final":       x_ln,
    }
    print(f"{'stage':15s} {'onnx_shape':20s} {'torch_shape':20s} {'max|Δ|':>14s} {'mean':>12s}")
    print("-" * 85)
    for lbl, nm in TAPS:
        yo = outs[nm]
        yt = tstages[lbl].cpu().numpy()
        if yo.shape != yt.shape:
            print(f"{lbl:15s} {str(yo.shape):20s} {str(yt.shape):20s}  SHAPE MISMATCH")
            continue
        d = np.abs(yo - yt)
        print(f"{lbl:15s} {str(yo.shape):20s} {str(yt.shape):20s}   {d.max():.4e}   {d.mean():.4e}")


if __name__ == "__main__":
    main()
