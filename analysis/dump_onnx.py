"""Dump ONNX graph structure and initializers for the 4 Supertonic-2 models.

Outputs per model:
  analysis/dump/<name>.summary.txt   -- io, op histogram, param totals
  analysis/dump/<name>.inits.tsv     -- name, shape, dtype, nbytes for every initializer
  analysis/dump/<name>.nodes.tsv     -- op_type, name, inputs, outputs (first N)
"""
import os
import onnx
from onnx import numpy_helper
from collections import Counter

ONNX_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "onnx")
OUT_DIR = os.path.join(os.path.dirname(__file__), "dump")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    "duration_predictor",
    "text_encoder",
    "vector_estimator",
    "vocoder",
]


def shape_str(tv):
    dims = []
    for d in tv.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.dim_value > 0 else (d.dim_param or "?"))
    return dims


def dump_model(name):
    path = os.path.join(ONNX_DIR, f"{name}.onnx")
    print(f"Loading {path} ...")
    m = onnx.load(path)
    g = m.graph

    # --- IO ---
    ios = []
    ios.append(f"# {name}.onnx")
    ios.append(f"opset: {[(o.domain or 'ai.onnx', o.version) for o in m.opset_import]}")
    ios.append(f"ir_version: {m.ir_version}   producer: {m.producer_name} {m.producer_version}")
    ios.append("")
    ios.append("## inputs")
    for t in g.input:
        ios.append(f"  {t.name:40s}  {shape_str(t)}  {onnx.TensorProto.DataType.Name(t.type.tensor_type.elem_type)}")
    ios.append("## outputs")
    for t in g.output:
        ios.append(f"  {t.name:40s}  {shape_str(t)}  {onnx.TensorProto.DataType.Name(t.type.tensor_type.elem_type)}")

    # --- op histogram ---
    ops = Counter(n.op_type for n in g.node)
    ios.append("")
    ios.append(f"## op histogram (total {len(g.node)} nodes)")
    for op, c in ops.most_common():
        ios.append(f"  {op:25s} {c}")

    # --- initializers ---
    total_params = 0
    total_bytes = 0
    init_rows = ["name\tshape\tdtype\tnbytes\tnumel"]
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        numel = int(arr.size)
        nbytes = int(arr.nbytes)
        total_params += numel
        total_bytes += nbytes
        init_rows.append(f"{init.name}\t{list(arr.shape)}\t{arr.dtype}\t{nbytes}\t{numel}")

    ios.append("")
    ios.append(f"## params: {total_params:,}  ({total_bytes/1e6:.2f} MB)")

    with open(os.path.join(OUT_DIR, f"{name}.summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ios))
    with open(os.path.join(OUT_DIR, f"{name}.inits.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(init_rows))

    # --- node list ---
    node_rows = ["idx\top_type\tname\tinputs\toutputs"]
    for i, n in enumerate(g.node):
        node_rows.append(f"{i}\t{n.op_type}\t{n.name}\t{list(n.input)}\t{list(n.output)}")
    with open(os.path.join(OUT_DIR, f"{name}.nodes.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(node_rows))

    print(f"  params={total_params:,}  nodes={len(g.node)}  file_MB={os.path.getsize(path)/1e6:.2f}")
    return total_params, len(g.node)


if __name__ == "__main__":
    grand_total = 0
    for name in MODELS:
        p, _ = dump_model(name)
        grand_total += p
    print(f"\nGRAND TOTAL params: {grand_total:,} ({grand_total/1e6:.2f} M)")
