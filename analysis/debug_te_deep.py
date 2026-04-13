"""Deep stage debug for text_encoder main stack."""
import os, sys, numpy as np, torch, onnx, onnxruntime as ort
from onnx import helper, TensorProto
sys.path.insert(0, os.path.dirname(__file__))
from torch_text_encoder import TextEncoder, load_text_encoder_weights

ONNX = os.path.join(os.path.dirname(__file__), "..", "assets", "onnx", "text_encoder.onnx")


def tap_taps():
    return [
        ("char_emb_masked",  "/text_encoder/text_embedder/Mul_output_0"),
        ("convnext_0_out",   "/text_encoder/convnext/convnext.0/Mul_3_output_0"),
        ("convnext_5_out",   "/text_encoder/convnext/convnext.5/Mul_3_output_0"),
        ("attn_enc_0_out",   "/text_encoder/attn_encoder/attn_layers.0/Mul_6_output_0"),  # may not exist; try others
    ]


def main():
    np.random.seed(0); torch.manual_seed(0)
    B, T = 2, 35
    text_ids = np.random.randint(0, 162, size=(B, T)).astype(np.int64)
    style_ttl = np.random.randn(B, 50, 256).astype(np.float32) * 0.3
    text_mask = np.zeros((B, 1, T), dtype=np.float32); text_mask[0,0,:30]=1; text_mask[1,0,:]=1

    # discover some attn_encoder outputs
    m = onnx.load(ONNX)
    attn_ln_outs = []
    for n in m.graph.node:
        if n.op_type == 'Mul' and '/text_encoder/attn_encoder/' in n.name and 'norm_layers' in n.name:
            # probably a mask Mul after norm
            attn_ln_outs.append(n.output[0])
        if n.op_type == 'Add' and '/text_encoder/attn_encoder/' == n.name[:len('/text_encoder/attn_encoder/')] and 'Add' in n.name.rsplit('/',1)[-1]:
            attn_ln_outs.append(n.output[0])

    # Keep 10 unique taps spread across layers
    attn_ln_outs = sorted(set(attn_ln_outs))[:20]
    print('taps:', attn_ln_outs[:10])

    m2 = onnx.load(ONNX)
    # drop attn_layers.0 Add taps (some are int64 size computations)
    keep = [t for t in attn_ln_outs if '/attn_layers.' not in t]
    names_to_tap = [
        "/text_encoder/text_embedder/Mul_output_0",
        "/text_encoder/convnext/convnext.0/Mul_3_output_0",
        "/text_encoder/convnext/convnext.5/Mul_3_output_0",
    ] + keep[:10] + [
        "/text_encoder/Add_output_0",  # final main output
    ]
    for nm in names_to_tap:
        m2.graph.output.append(helper.make_tensor_value_info(nm, TensorProto.FLOAT, None))
    tmp = os.path.join(os.path.dirname(__file__), "_te_deep.onnx")
    onnx.save(m2, tmp)
    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    outs = dict(zip(out_names, sess.run(None, {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask})))

    model = TextEncoder(); load_text_encoder_weights(model, ONNX); model.eval()
    # run torch step by step, matching tap points
    te = model.text_encoder
    with torch.no_grad():
        tids = torch.from_numpy(text_ids); tm = torch.from_numpy(text_mask)
        e = te.char_embedder(tids).transpose(1, 2) * tm
        print("char_emb_masked diff:", np.abs(e.numpy() - outs["/text_encoder/text_embedder/Mul_output_0"]).max())
        x = e
        for i, blk in enumerate(te.convnext):
            x = blk(x, tm)
            nm = f"/text_encoder/convnext/convnext.{i}/Mul_3_output_0"
            if nm in outs:
                d = np.abs(x.numpy() - outs[nm]).max()
                print(f"  convnext.{i}.out diff: {d:.4e}")
        x_after_conv = x
        for i, lyr in enumerate(te.attn_layers):
            x = lyr(x, tm)
            print(f"  attn_layer.{i}.out torch-range: [{x.min():.3f}, {x.max():.3f}]")
        x = (x + x_after_conv) * tm
        d = np.abs(x.numpy() - outs["/text_encoder/Add_output_0"]).max()
        print(f"main final diff: {d:.4e}")


if __name__ == "__main__":
    main()
