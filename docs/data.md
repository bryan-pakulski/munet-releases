# Data loading

`munet.data` turns a dataset into batches of NumPy arrays. It handles shuffling, background loading in worker
threads, prefetching the next batches while the GPU runs, and sharding across GPUs or processes.

```python
from munet.data import ArrayDataset, DataLoader

loader = DataLoader(ArrayDataset(images, labels), batch_size=64, shuffle=True, workers=4)
for epoch in range(10):
    for x, y in loader:              # x: [64, ...] float32, y: [64] int64 (whatever your arrays hold)
        loss = train_step(x, y)
```

## Datasets

**In-memory arrays:** `ArrayDataset(*arrays)`. Sample `i` is `(a[i], b[i], ...)`. Without a transform, batches are
sliced from the arrays in one step, which is the fastest option. `np.load(..., mmap_mode="r")` arrays work too and
stream from disk.

**Anything else:** a class with `__len__` and `__getitem__(i)` that returns one sample. A sample is an array, a
number, or a tuple / list / dict of them:

```python
from pathlib import Path
import numpy as np
from PIL import Image

class ImageFolder:
    """root/<class name>/<image>.jpg -> (CHW float32 image, class index)"""
    def __init__(self, root, size=224):
        self.classes = sorted(p.name for p in Path(root).iterdir() if p.is_dir())
        self.items = [(f, i) for i, c in enumerate(self.classes) for f in sorted((Path(root) / c).glob("*.jpg"))]
        self.size = size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        image = Image.open(path).convert("RGB").resize((self.size, self.size))
        x = np.asarray(image, np.float32).transpose(2, 0, 1) / 255.0
        return x, label

loader = DataLoader(ImageFolder("data/train"), batch_size=64, shuffle=True, workers=8)
```

Dict samples give dict batches: `{"image": ..., "boxes": ...}` → `batch["image"]` is `[B, ...]`.

## Augmentation

Per-sample transforms run in the worker threads:

```python
def augment(sample):
    image, label = sample
    if np.random.random() < 0.5:
        image = image[:, :, ::-1]                      # horizontal flip
    return np.ascontiguousarray(image), label

train = ArrayDataset(images, labels, transform=augment)
```

For a custom dataset, apply the transform inside `__getitem__`.

## DataLoader options

| Option | Default | Meaning |
|---|---|---|
| `batch_size` | 1 | samples per batch |
| `shuffle` | False | new order every epoch, seeded by `seed + epoch` (reproducible) |
| `drop_last` | **True** | drop the last partial batch: compiled programs have fixed shapes |
| `workers` | 0 | threads loading samples in parallel |
| `prefetch` | 2 | batches prepared ahead while the GPU works (0 = load on demand) |
| `collate` | stack | list of samples → batch; replace for variable-size data (pad to a fixed size there) |
| `shard` | None | `(index, count)`: iterate only this consumer's share of each epoch |
| `seed` | 0 | shuffle seed |

- **Epochs.** Each `for` loop over the loader is one epoch with a new shuffle order. Call `loader.set_epoch(n)` to
  reproduce epoch `n`'s order, for example after resuming from a checkpoint.
- **Batch count.** `len(loader)` is the number of batches per epoch.

## Making it fast

- **Keep the GPU fed.** Leave `prefetch` at 2 or more, so loading batch N+1 overlaps the GPU running batch N.
- **Workers are threads.** They pay off when samples are loaded by code that releases the GIL: file reads, Pillow
  decode/resize, NumPy. Pure-Python per-pixel code doesn't speed up with threads. Vectorize it with NumPy, or
  preprocess once and save arrays.
- **Fastest path:** decode once, `np.save` the arrays, then `ArrayDataset(np.load("x.npy", mmap_mode="r"), ...)`.
- **Don't read results every step.** Calling `loss.item()` each step makes the CPU wait for the GPU. Read it every
  N steps.

## Several GPUs or processes

With `mu.DataParallel`, use **one** loader with the global batch size; DataParallel splits each batch across the
GPUs.

When each GPU or process has its own program (separate processes, or one inference worker per GPU), shard the data:

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True, seed=0, shard=(rank, world_size))
```

Shards are disjoint, use the same shuffled order, and have the same number of batches. Use the same `seed` in every
process. See [multi-gpu.md](multi-gpu.md).

## Variable-size data

Programs have static shapes, so every batch must have the same shape. Pad or resize to fixed sizes in `__getitem__`
or `collate`. For detection, use a fixed maximum number of boxes plus a mask. Using several input sizes (e.g.
multi-scale training) means compiling one step per size and picking the function that matches each batch.
