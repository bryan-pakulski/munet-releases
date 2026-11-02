# Multiple GPUs

MuNet uses every Vulkan GPU in the machine, whatever the vendor. You can mix NVIDIA, AMD and Intel cards in one
training run.

- [Train on several GPUs](#train-on-several-gpus)
- [GPUs of different size and speed](#gpus-of-different-size-and-speed) (e.g. RTX 3090 + 3060 + 4060)
- [Data loading for several GPUs](#data-loading-for-several-gpus)
- [Inference on several GPUs](#inference-on-several-gpus)
- [Pick which GPU "vulkan" means](#pick-which-gpu-vulkan-means)

## Train on several GPUs

```python
import munet as mu
from munet.data import ArrayDataset, DataLoader

model = MyNet()
optimizer = mu.optim.AdamW(model.parameters(), lr=1e-3)

def loss_fn(images, labels):                     # must return the batch MEAN loss
    return mu.nn.functional.cross_entropy(model(images), labels)

dp = mu.DataParallel(loss_fn, model.parameters(), optimizer,
                     ["vulkan:0", "vulkan:1"], precision="mixed")

loader = DataLoader(ArrayDataset(images, labels), batch_size=256, shuffle=True)
for epoch in range(10):
    for x, y in loader:
        loss = dp.step(x, y)                     # Python float, batch-mean loss
```

How a step works:

1. The batch is split on dim 0 into one shard per GPU.
2. Every GPU computes its shard's gradients in parallel.
3. The gradients are averaged, weighted by shard size.
4. One optimizer step is applied, and all GPUs receive the new weights.

The result is the same as one full-batch step on a single GPU, for any split. Training code is unchanged: `model`
holds the trained weights afterwards, so evaluate, checkpoint and export as usual.

- `mu.devices()` lists the GPUs; `"vulkan:N"` is the N-th entry.
- The same device may appear twice (two replicas on one GPU).
- `loss_fn` must return the batch mean (`reduction="mean"`, the default for every loss).

Runnable example: [examples/python/train_multi_gpu.py](../examples/python/train_multi_gpu.py).

## GPUs of different size and speed

`split=` decides how many rows of each batch every GPU gets. `max_rows=` caps it per GPU.

| `split=` | Rows per GPU |
|---|---|
| `"even"` (default) | equal |
| `"memory"` | proportional to each GPU's memory (24 GB : 12 GB : 8 GB → 24 : 12 : 8) |
| `"auto"` | proportional to each GPU's measured speed, re-balanced over the first steps |
| `[3, 2, 1]` | fixed weights, one per device |

| `max_rows=` | Effect |
|---|---|
| `[None, None, 48]` | GPU 2 never gets more than 48 rows per step; its excess goes to the other GPUs in proportion to their weights |

### Example: RTX 3090 (24 GB) + RTX 3060 (12 GB) + RTX 4060 (8 GB)

These cards differ in two ways. The 3090 is roughly twice as fast as the other two, and the 4060 has a third of
the 3090's memory. Use one of these setups.

**Simplest: split by memory.** Each card gets rows in proportion to its memory:

```python
devices = ["vulkan:0", "vulkan:1", "vulkan:2"]           # check the order with mu.devices()
dp = mu.DataParallel(loss_fn, model.parameters(), optimizer, devices,
                     precision="mixed", split="memory")   # a 512-row batch -> 279 / 140 / 93 rows
```

`dp.counts` shows the rows each GPU got in the last step. Weights, gradients and optimizer state take the same memory
on every card, so the smallest card fills up first. If it runs out, add a cap for it (next setup).

**Fastest: split by speed, cap by memory.** `"auto"` gives the 3090 about half of every batch. The caps keep the
smaller cards inside their memory whatever the speed measurement says:

```python
dp = mu.DataParallel(loss_fn, model.parameters(), optimizer, devices,
                     precision="mixed", split="auto", max_rows=[None, 160, 96])   # example caps: measure yours below
```

**Fixed: you know the ratio.** This avoids the few recompiles that `"auto"` does while it settles:

```python
dp = mu.DataParallel(..., devices, split=[2, 1, 1], max_rows=[None, 160, 96])
```

The same from the command line: `python examples/python/train_multi_gpu.py --split memory --max-rows 0,160,96`
(`0` = no cap).

### Find the caps for your model

The memory a GPU needs grows linearly with its rows. `dp.stats()` reports each GPU's rows, step time and device
memory (`arena_bytes`, which includes weights, gradients and optimizer state):

```python
dp = mu.DataParallel(loss_fn, model.parameters(), optimizer, devices, split="even")
dp.step(x[:48], y[:48]); a = dp.stats()        # 16 rows per GPU
dp.step(x[:96], y[:96]); b = dp.stats()        # 32 rows per GPU
for s0, s1 in zip(a, b):
    per_row = (s1["arena_bytes"] - s0["arena_bytes"]) / (s1["rows"] - s0["rows"])
    fixed = s0["arena_bytes"] - per_row * s0["rows"]
    print(s0["device"], f"{per_row / 2**20:.1f} MiB/row, {fixed / 2**20:.0f} MiB fixed")
```

Set `max_rows` to `(memory_you_can_use - fixed) / per_row`. Leave 10-20% of the card free for the driver, the
display and anything else running on it.

### Rules and limits

- **The global batch must fit.** It cannot exceed the sum of the caps, and must have at least one row per GPU.
  Otherwise `dp.step` raises `ValueError`.
- **Shapes are static.** Each new shard size compiles a program for that GPU once. `"auto"` moves in steps of 1/16
  of the batch and stops after 4 moves, so expect a few slow steps at the start and none after.
- **Out of memory.** If a GPU runs out of memory, the error names the replica, device and row count. Lower that
  GPU's weight or `max_rows`.
- **Gradient exchange.** Gradients are averaged on the host, so throughput scales best when compute per step is
  large compared with the model size. Use big batches, and convolutional or attention models rather than tiny MLPs.
- **One machine only.** DataParallel covers the GPUs in one machine. Training across machines uses the training swarm
  ([swarm.md](swarm.md)).

## Data loading for several GPUs

- **With DataParallel, use one loader** that yields global batches; DataParallel splits them. Size `batch_size` for
  all GPUs together.
- **For one process per GPU** (or one inference worker per GPU), give each worker its own share of every epoch with
  `shard=(index, count)`:

  ```python
  loader = DataLoader(dataset, batch_size=64, shuffle=True, seed=0, shard=(gpu_index, gpu_count))
  ```

  All shards use the same shuffled order, take disjoint rows, and get the same number of batches. Call
  `loader.set_epoch(epoch)` (or just iterate again) for a new order.

See [data.md](data.md) for the loader itself.

## Inference on several GPUs

Load one program per GPU and feed each from its own thread. Calls release the GIL, so the GPUs run concurrently:

```python
from concurrent.futures import ThreadPoolExecutor
from munet.interop import from_onnx

gpus = [f"vulkan:{i}" for i in range(len(mu.devices()))]
programs = [from_onnx("model.onnx", device=d, precision="fp16", input_shapes={"images": (8, 3, 640, 640)}) for d in gpus]

def worker(i):
    loader = DataLoader(dataset, batch_size=8, shard=(i, len(gpus)))
    for batch in loader:
        out = programs[i](batch).numpy()
        ...

with ThreadPoolExecutor(len(gpus)) as pool:
    list(pool.map(worker, range(len(gpus))))
```

For GPUs of different speed, use a shared queue of batches instead of fixed shards. Each worker then takes the next
batch when it is free, and faster GPUs do more of the work. See
[examples/python/infer_multi_gpu.py](../examples/python/infer_multi_gpu.py).

From C, C++ and Rust, load the `.mnet` once per GPU (`options.device = N`) and use one thread per program.

## Pick which GPU "vulkan" means

`device="vulkan"` means GPU 0 unless `MUNET_VULKAN_DEVICE=N` is set. This moves a whole script or test suite to
another GPU without code changes:

```bash
MUNET_VULKAN_DEVICE=1 python train.py
```
