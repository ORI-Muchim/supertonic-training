"""Summarize named parameters (filter out /Constant_*) and group by prefix."""
import os
import csv
from collections import defaultdict

DUMP = os.path.join(os.path.dirname(__file__), "dump")

MODELS = ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]


def load_inits(name):
    rows = []
    with open(os.path.join(DUMP, f"{name}.inits.tsv"), encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append(row)
    return rows


def is_real_param(name):
    # PyTorch exports real weights with dotted names (module.submod.weight)
    # ONNX constants typically start with "/" or have "_output_0"
    if name.startswith("/"):
        return False
    if "Constant" in name and "output" in name:
        return False
    return True


def group_by_prefix(rows, depth=3):
    groups = defaultdict(lambda: {"count": 0, "numel": 0, "examples": []})
    for r in rows:
        if not is_real_param(r["name"]):
            continue
        parts = r["name"].split(".")
        prefix = ".".join(parts[:depth])
        groups[prefix]["count"] += 1
        groups[prefix]["numel"] += int(r["numel"])
        if len(groups[prefix]["examples"]) < 2:
            groups[prefix]["examples"].append(f"{r['name']} {r['shape']}")
    return groups


def main():
    with open(os.path.join(DUMP, "param_groups.md"), "w", encoding="utf-8") as fout:
        for name in MODELS:
            rows = load_inits(name)
            real = [r for r in rows if is_real_param(r["name"])]
            total = sum(int(r["numel"]) for r in real)
            fout.write(f"\n## {name}.onnx  ({len(real)} real params, {total:,} elements)\n\n")
            for depth in (3, 4, 5):
                grp = group_by_prefix(rows, depth=depth)
                fout.write(f"### grouped by prefix depth={depth}\n\n")
                fout.write("| prefix | #tensors | numel | example |\n|---|---:|---:|---|\n")
                for k, v in sorted(grp.items(), key=lambda kv: -kv[1]["numel"]):
                    fout.write(f"| `{k}` | {v['count']} | {v['numel']:,} | `{v['examples'][0]}` |\n")
                fout.write("\n")
            # raw list of unique top-level prefixes
            top = set(r["name"].split(".")[0] for r in real)
            fout.write(f"top-level roots: {sorted(top)}\n")

    print("wrote", os.path.join(DUMP, "param_groups.md"))


if __name__ == "__main__":
    main()
