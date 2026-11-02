"""Find the first ONNX node whose output MuNet computes differently from ONNX Runtime.

    python examples/bench/find_mismatch.py MODEL_NAME [--device cpu|vulkan:N] [--precision fp32] [--threshold 0.9999]

Every intermediate float tensor becomes a graph output; both runtimes evaluate the extended graph on the suite's inputs
and the nodes are compared in graph order (cosine similarity). Prints the first mismatches with the node's op and inputs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import select  # noqa: E402
from run import build_inputs, fetch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--threshold", type=float, default=0.9999)
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()
    import onnx
    import onnxruntime as ort
    from onnx import helper, shape_inference, TensorProto
    from munet.interop import from_onnx

    model = select(names=[args.model])[0]
    path = fetch(model.release, model.file)
    graph = onnx.load(str(path))
    arrays, shapes = build_inputs(model, graph)
    # static shapes, inferred types for every intermediate
    for vi in graph.graph.input:
        if vi.name in shapes:
            for d, n in zip(vi.type.tensor_type.shape.dim, shapes[vi.name]):
                d.Clear(); d.dim_value = n
    inferred = shape_inference.infer_shapes(graph)
    known = {vi.name: vi for vi in list(inferred.graph.value_info) + list(inferred.graph.output)}
    existing = {o.name for o in graph.graph.output}
    extra = []
    for node in graph.graph.node:
        for out in node.output:
            vi = known.get(out)
            if out and out not in existing and vi is not None and vi.type.tensor_type.elem_type == TensorProto.FLOAT \
                    and all(d.HasField("dim_value") for d in vi.type.tensor_type.shape.dim):
                extra.append(vi)
    graph.graph.output.extend(extra)
    names = [o.name for o in graph.graph.output]
    print(f"{len(extra)} intermediate outputs", flush=True)
    ref = ort.InferenceSession(graph.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, arrays)
    program = from_onnx(graph, device=args.device, precision=args.precision, input_shapes=shapes)
    outs = program(*arrays.values())
    outs = outs if isinstance(outs, (list, tuple)) else [outs]
    outs = {n: (o.numpy() if hasattr(o, "numpy") else np.asarray(o)) for n, o in zip(names, outs)}
    produced = {out: node for node in graph.graph.node for out in node.output}
    shown = 0
    for node in graph.graph.node:
        for out in node.output:
            if out not in outs:
                continue
            a = outs[out].astype(np.float64).ravel(); b = ref[names.index(out)].astype(np.float64).ravel()
            if a.shape != b.shape:
                print(f"SHAPE {node.op_type} {node.name}: {outs[out].shape} vs {ref[names.index(out)].shape}"); shown += 1
                continue
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            cos = a @ b / (na * nb) if na > 0 and nb > 0 else (1.0 if np.allclose(a, b) else 0.0)
            if cos < args.threshold:
                ins = [f"{i}<{produced[i].op_type}>" if i in produced else i for i in node.input]
                attrs = {x.name: helper.get_attribute_value(x) for x in node.attribute}
                print(f"cos {cos:.6f} {node.op_type} {node.name} inputs={ins} attrs={attrs} max|d|={np.abs(a-b).max():.3g}")
                shown += 1
            if shown >= args.show:
                return
    print("no mismatch above threshold" if not shown else "")


if __name__ == "__main__":
    main()
