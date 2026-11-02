"""Train on every GPU in the machine with munet.DataParallel (any vendor mix: e.g. an NVIDIA and an Intel card).

    python examples/python/train_multi_gpu.py                          # all Vulkan GPUs, split by measured speed
    python examples/python/train_multi_gpu.py --devices vulkan:0,vulkan:1 --split even
    python examples/python/train_multi_gpu.py --split memory --max-rows 0,0,48    # e.g. 24 / 12 / 8 GB cards

Each step splits the batch across the GPUs, computes gradients on all of them in parallel, averages them and applies
one optimizer step: the same result as one full-batch step on one GPU. The model and data are the ones from
train_classifier.py.
"""
import argparse
import time

import numpy as np

import munet as mu
from munet.data import ArrayDataset, DataLoader

from train_classifier import Net, make_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", help="comma-separated, default: every Vulkan GPU")
    ap.add_argument("--split", default="auto",
                    help='"auto" (by measured speed), "memory" (by GPU memory), "even", or weights like 3,2,1')
    ap.add_argument("--max-rows", help="per-GPU cap on rows per step, e.g. 0,0,48 (0 = no cap)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256, help="global batch, split across the GPUs")
    args = ap.parse_args()

    devices = args.devices.split(",") if args.devices else [f"vulkan:{i}" for i in range(len(mu.devices()))]
    split = args.split if args.split in ("auto", "even", "memory") else [float(w) for w in args.split.split(",")]
    max_rows = [int(m) or None for m in args.max_rows.split(",")] if args.max_rows else None
    print("training on", ", ".join(f"{d} ({mu.devices()[int(d.split(':')[1])]})" for d in devices))

    x, y = make_dataset(8192, seed=0)
    loader = DataLoader(ArrayDataset(x, y), batch_size=args.batch, shuffle=True)

    model = Net(np.random.default_rng(0))
    optimizer = mu.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    def loss_fn(images, labels):  # mean over the batch rows; DataParallel weights each GPU's shard by its size
        return mu.nn.functional.cross_entropy(model(images), labels)

    dp = mu.DataParallel(loss_fn, model.parameters(), optimizer, devices, precision="mixed", split=split, max_rows=max_rows)

    for epoch in range(args.epochs):
        t = time.perf_counter()
        for images, labels in loader:
            loss = dp.step(images, labels)  # Python float: the batch-mean loss
        elapsed = time.perf_counter() - t
        print(f"epoch {epoch}: loss {loss:.4f}  {len(loader) * args.batch / elapsed:,.0f} images/s")
        for s in dp.stats():  # rows each GPU got and its memory: use this to size --max-rows
            print(f"    {s['device']}: {s['rows']} rows, {s['seconds'] * 1e3:.1f} ms, {s['arena_bytes'] / 2**20:.0f} MiB")

    # model's parameters hold the trained weights: evaluate, checkpoint or export as with single-GPU training.
    model.eval()
    predict = mu.compile(model, device=devices[0])
    val_x, val_y = make_dataset(512, seed=1)
    print(f"val acc {(predict(val_x).numpy().argmax(1) == val_y).mean():.3f}")


if __name__ == "__main__":
    main()
