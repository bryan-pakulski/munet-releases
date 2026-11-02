"""Matrix of bench-suite results: latest results file per device (or the files given), one row per model.

    python examples/bench/report.py [results/<device>/<stamp>.json ...] [--markdown report.md]

Each cell: status and GPU ms per precision (PASS / WARN / FAIL / ERR), then TensorRT ms where measured.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ("fp32", "mixed", "fp16", "int8")
MARK = {"PASS": "ok", "WARN": "warn", "FAIL": "FAIL", "ERROR": "ERR"}


def latest(results):
    files = []
    for d in sorted(Path(results).glob("*")):
        runs = sorted(d.glob("*.json"))
        if runs:
            files.append(runs[-1])
    return files


def merged(results):
    """One run per device: the newest result for every (model, precision) across that device's results files."""
    runs = []
    for d in sorted(Path(results).glob("*")):
        files = sorted(d.glob("*.json"))
        if not files:
            continue
        run = json.loads(files[-1].read_text())
        by, trt = {}, {}
        for f in files:  # oldest first, newer results overwrite
            data = json.loads(f.read_text())
            for r in data["results"]:
                by[(r["model"], r["precision"])] = r
            for t in data.get("tensorrt", []):
                trt[(t["model"], t["precision"])] = t
        run["results"], run["tensorrt"] = list(by.values()), list(trt.values())
        runs.append(run)
    return runs


def render(files=None, runs=None):
    runs = runs if runs is not None else [json.loads(Path(f).read_text()) for f in files]
    models = []
    for run in runs:
        for r in run["results"]:
            if r["model"] not in models:
                models.append(r["model"])
    lines = []
    for run in runs:
        by = {(r["model"], r["precision"]): r for r in run["results"]}
        trt = {(t["model"], t["precision"]): t for t in run.get("tensorrt", [])}
        lines.append(f"### {run['device']} (MuNet {run['munet']}, {run['date']})\n")
        head = "| model | " + " | ".join(ORDER) + (" | TRT fp32 | TRT fp16 | TRT int8" if trt else "") + " |"
        lines += [head, "|" + "---|" * (head.count("|") - 1)]
        counts = {}
        for m in models:
            cells = []
            for p in ORDER:
                r = by.get((m, p))
                if not r:
                    cells.append("")
                    continue
                counts[r["status"]] = counts.get(r["status"], 0) + 1
                ms = f" {r['gpu_ms']:.2f} ms" if "gpu_ms" in r else ""
                worst = min((x.get("cos", 1.0) for x in r.get("metrics", [])), default=None)
                detail = f" (cos {worst:.6f})" if r["status"] in ("WARN", "FAIL") and worst is not None else ""
                err = f" `{r.get('error', '')[:60]}`" if r["status"] == "ERROR" else ""
                cells.append(f"{MARK[r['status']]}{ms}{detail}{err}")
            if trt:
                for p in ("fp32", "fp16", "int8"):
                    t = trt.get((m, p))
                    cells.append(f"{t['gpu_ms']:.2f} ms" if t and "gpu_ms" in t else ("err" if t else ""))
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + "\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--markdown")
    args = ap.parse_args()
    text = render(args.files) if args.files else render(runs=merged(args.results))
    print(text)
    if args.markdown:
        Path(args.markdown).write_text(text + "\n")


if __name__ == "__main__":
    main()
