"""Stage-by-stage compare DP PyTorch vs ONNX."""
import os, sys, numpy as np, torch, onnx, onnxruntime as ort
from onnx import helper, TensorProto

sys.path.insert(0, os.path.dirname(__file__))
from torch_duration_predictor import DurationPredictor, load_dp_weights, causal_pad_rep

ONNX = os.path.join(os.path.dirname(__file__), "..", "assets", "onnx", "duration_predictor.onnx")

TAPS = [
    ("char_emb_masked",  "/sentence_encoder/text_embedder/Mul_output_0"),
    ("with_sent_token",  "/sentence_encoder/Concat_1_output_0"),
    ("convnext_0_out",   "/sentence_encoder/convnext/convnext.0/Mul_3_output_0"),
    ("convnext_5_out",   "/sentence_encoder/convnext/convnext.5/Mul_3_output_0"),
    ("attn0_out",        "/sentence_encoder/attn_encoder/Mul_1_output_0"),
    ("final_add",        "/sentence_encoder/Add_output_0"),
    ("proj_out",         "/sentence_encoder/proj_out/Mul_output_0"),
    ("pre_act",          "/predictor/layers.0/Gemm_output_0"),
    ("post_act",         "/predictor/activation/PRelu_output_0"),
    ("pre_exp",          "/predictor/layers.1/Gemm_output_0"),
    ("final",            "duration"),
]


def run_onnx(x1, x2, x3):
    m = onnx.load(ONNX)
    for lbl, out_name in TAPS:
        m.graph.output.append(helper.make_tensor_value_info(out_name, TensorProto.FLOAT, None))
    tmp = os.path.join(os.path.dirname(__file__), "_dp_tap.onnx")
    onnx.save(m, tmp)
    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(names, {"text_ids": x1, "style_dp": x2, "text_mask": x3})
    return dict(zip(names, outs))


def torch_stages(model, text_ids, style_dp, text_mask):
    stages = {}
    se = model.sentence_encoder
    B, T = text_ids.shape
    e = se.char_embedder(text_ids).transpose(1, 2) * text_mask
    stages["char_emb_masked"] = e
    st = se.sentence_token.expand(B, -1, 1)
    x = torch.cat([st, e], dim=-1)
    mask = torch.cat([torch.ones(B, 1, 1), text_mask], dim=-1)
    stages["with_sent_token"] = x
    for i, blk in enumerate(se.convnext):
        x = blk(x, mask)
        if i == 0: stages["convnext_0_out"] = x
        if i == 5: stages["convnext_5_out"] = x
    x_after_convnext = x
    for i, lyr in enumerate(se.attn_layers):
        x = lyr(x, mask)
        if i == 0: stages["attn0_out"] = x
    x = x + x_after_convnext
    stages["final_add"] = x
    sent = x[:, :, :1]
    mask1 = mask[:, :, :1]
    sent = se.proj_out(sent) * mask1
    stages["proj_out"] = sent

    sent_flat = sent.reshape(B, -1)
    style_flat = style_dp.reshape(B, -1)
    h = torch.cat([sent_flat, style_flat], dim=-1)
    h = model.fc1(h); stages["pre_act"] = h
    h = model.act(h); stages["post_act"] = h
    h = model.fc2(h); stages["pre_exp"] = h
    stages["final"] = torch.exp(h).squeeze(-1)
    return stages


def main():
    np.random.seed(0); torch.manual_seed(0)
    B, T = 2, 30
    text_ids = np.random.randint(0, 162, size=(B, T)).astype(np.int64)
    style_dp = np.random.randn(B, 8, 16).astype(np.float32) * 0.2
    text_mask = np.zeros((B, 1, T), dtype=np.float32)
    text_mask[0, 0, :25] = 1.0
    text_mask[1, 0, :T] = 1.0

    m = DurationPredictor(); load_dp_weights(m, ONNX); m.eval()
    with torch.no_grad():
        stg = torch_stages(m, torch.from_numpy(text_ids), torch.from_numpy(style_dp), torch.from_numpy(text_mask))
    outs = run_onnx(text_ids, style_dp, text_mask)

    print(f"{'stage':20s} {'onnx_shape':20s} {'torch_shape':20s} {'max|Δ|':>12s} {'mean|Δ|':>12s}")
    print("-" * 85)
    for lbl, nm in TAPS:
        yo = outs[nm]
        yt = stg[lbl].cpu().numpy()
        if yo.shape != yt.shape:
            print(f"{lbl:20s} {str(yo.shape):20s} {str(yt.shape):20s}   SHAPE MISMATCH")
            continue
        d = np.abs(yo - yt)
        print(f"{lbl:20s} {str(yo.shape):20s} {str(yt.shape):20s}   {d.max():.4e}   {d.mean():.4e}")


if __name__ == "__main__":
    main()
