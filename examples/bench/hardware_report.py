"""Markdown speed tables for docs/tested-hardware.md from bench results (examples/bench/run.py, optionally --trt).

    python examples/bench/hardware_report.py --baseline "RTX 3060" \\
        "RTX 3060=examples/bench/results/nvidia-geforce-rtx-3060" "T4=results/aws/g4dn.xlarge/bench" ...

Each argument is LABEL=PATH, where PATH is a result JSON or a directory (its newest JSON, searched recursively). Prints:
1. MuNet GPU time per model and GPU (ms, batch 1), with the geometric mean;
2. speed relative to the baseline GPU (baseline time / GPU time: above 1 is faster);
3. MuNet against TensorRT on the same GPU (TensorRT time / MuNet time: 100% is parity), where the run has --trt.
Times are minimum GPU compute times (MuNet: GPU timestamps; TensorRT: trtexec --noDataTransfers).
"""
import argparse
import json
import math
from pathlib import Path


def load(path):
    p = Path(path)
    if p.is_dir():
        files = sorted(p.rglob("*.json"), key=lambda f: f.stat().st_mtime)
        if not files:
            raise SystemExit(f"no result JSON under {p}")
        p = files[-1]
    run = json.loads(p.read_text())
    return run if isinstance(run, dict) else {"results": run}


def times(run, key, precision):
    return {r["model"]: r["gpu_ms"] for r in run.get(key, []) if r.get("precision") == precision and r.get("gpu_ms")}


def geomean(values):
    values = [v for v in values if v and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def table(header, rows):
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    line = lambda cells: "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"
    return "\n".join([line(header), "| " + " | ".join("-" * w for w in widths) + " |"] + [line(r) for r in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="LABEL=PATH (result JSON or directory)")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--baseline", help="label of the baseline GPU for the relative table (default: the first)")
    ap.add_argument("--trt-mean-exclude", default="", help="comma-separated models kept in the TensorRT table but left "
                    "out of its geometric mean (outliers such as a TensorRT tactic failure)")
    args = ap.parse_args()
    labels, munet, trt = [], {}, {}
    for spec in args.runs:
        label, _, path = spec.partition("=")
        run = load(path)
        labels.append(label)
        munet[label] = times(run, "results", args.precision)
        trt[label] = times(run, "tensorrt", args.precision)
    models = sorted({m for t in munet.values() for m in t})
    excluded = {m for m in args.trt_mean_exclude.split(",") if m}
    mean_models = [m for m in models if m not in excluded]
    note = f" (excluding {', '.join(sorted(excluded))})" if excluded else ""
    base = args.baseline or labels[0]
    fmt = lambda v, f="{:.2f}": f.format(v) if v is not None else "–"

    rows = [[m] + [fmt(munet[l].get(m)) for l in labels] for m in models]
    rows.append(["**geometric mean**"] + [fmt(geomean([munet[l].get(m) for m in models])) for l in labels])
    print(f"### MuNet {args.precision}: GPU ms per inference (batch 1, lower is better)\n")
    print(table(["Model"] + labels, rows))

    rel = lambda l, m: munet[base][m] / munet[l][m] if m in munet[base] and m in munet[l] else None
    rows = [[m] + [fmt(rel(l, m), "{:.2f}×") for l in labels] for m in models]
    rows.append(["**geometric mean**"] + [fmt(geomean([rel(l, m) for m in models]), "{:.2f}×") for l in labels])
    print(f"\n### Speed relative to {base} (higher is faster; {base} = 1.00×)\n")
    print(table(["Model"] + labels, rows))

    with_trt = [l for l in labels if trt[l]]
    if with_trt:
        pct = lambda l, m: trt[l][m] / munet[l][m] if m in trt[l] and m in munet[l] else None
        rows = [[m] + [f"{fmt(munet[l].get(m))} / {fmt(trt[l].get(m))} ({fmt(pct(l, m) and 100 * pct(l, m), '{:.0f}%')})"
                       for l in with_trt] for m in models]
        rows.append([f"**geometric mean**{note}"] + [fmt(geomean([pct(l, m) for m in mean_models]) and 100 * geomean([pct(l, m) for m in mean_models]), "{:.0f}%")
                                                     for l in with_trt])
        print(f"\n### MuNet vs TensorRT {args.precision} on the same GPU: MuNet ms / TensorRT ms (MuNet speed as % of TensorRT)\n")
        print(table(["Model"] + with_trt, rows))


if __name__ == "__main__":
    main()
