# Training

MuNet training looks like PyTorch, with one difference: you put the training step in a function and compile it.
The first call traces the function and builds one GPU program for forward, loss, backward and optimizer update.
Every later call replays that program, and weights and optimizer state stay on the GPU between calls.

Complete runnable example: [examples/python/train_classifier.py](../examples/python/train_classifier.py). It covers
data loading, augmentation, training, evaluation, an LR schedule, checkpoint/resume, and export to `.mnet` and ONNX.

- [The training loop](#the-training-loop)
- [Models](#models)
- [Losses](#losses)
- [Optimizers and learning-rate schedules](#optimizers-and-learning-rate-schedules)
- [Evaluation](#evaluation)
- [Checkpoints and resuming](#checkpoints-and-resuming)
- [Precision](#precision)
- [Gradient clipping, EMA, freezing layers](#gradient-clipping-ema-freezing-layers)
- [Use PyTorch weights](#use-pytorch-weights)
- [Export the trained model](#export-the-trained-model)
- [Rules of compiled functions](#rules-of-compiled-functions)

## The training loop

```python
import numpy as np
import munet as mu
from munet.data import ArrayDataset, DataLoader

model = MyNet()
optimizer = mu.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

@mu.compile(device="vulkan")                 # or "vulkan:1", "cpu"
def train_step(images, labels):
    optimizer.zero_grad()
    loss = mu.nn.functional.cross_entropy(model(images), labels)
    loss.backward()
    optimizer.step()
    return loss

loader = DataLoader(ArrayDataset(train_images, train_labels), batch_size=64, shuffle=True, workers=4)
for epoch in range(20):
    model.train()
    for images, labels in loader:
        loss = train_step(images, labels)    # NumPy in, lazy Result out
    print(epoch, loss.item())                # .item() waits for the GPU and reads the value back
```

- **Inputs** are NumPy arrays. Float arrays become float32, and integer/bool arrays (labels, masks) convert exactly
  up to 2^24.
- **Outputs** are `Result`s. `.item()` and `.numpy()` read them back and wait for the GPU, so don't call them every
  step if you care about speed. A `Result` is valid until the next call of the same function.
- **Return anything you want to watch:** `return loss, accuracy` gives a tuple of `Result`s.
- **The first call compiles.** It takes seconds, and kernels are cached on disk for the next run. Later calls take
  milliseconds.

## Models

Build models from `mu.nn` modules, or subclass `mu.nn.Module`:

```python
class MyNet(mu.nn.Module):
    def __init__(self, classes=10):
        self.features = mu.nn.Sequential(
            mu.nn.Conv2d(3, 32, 3, padding=1), mu.nn.BatchNorm2d(32), mu.nn.ReLU(), mu.nn.MaxPool2d(2),
            mu.nn.Conv2d(32, 64, 3, padding=1), mu.nn.BatchNorm2d(64), mu.nn.ReLU(),
            mu.nn.AdaptiveAvgPool2d(1), mu.nn.Flatten())
        self.head = mu.nn.Linear(64, classes)

    def forward(self, x):
        return self.head(self.features(x))
```

- **No `super().__init__()` needed.** Attributes that are modules, `mu.Parameter`s or lists/dicts of them are
  discovered automatically.
- **Deterministic initialization:** layers take `rng=np.random.default_rng(seed)`.
- **Custom weights** are `mu.Parameter(numpy_array)`. Non-trained state is `self.register_buffer(name, array)`.

| Group | Modules |
|---|---|
| Linear / conv | `Linear`, `Conv1d`, `Conv2d` (groups, dilation), `ConvTranspose1d`, `ConvTranspose2d`, `Embedding` |
| Normalization | `BatchNorm1d`, `BatchNorm2d`, `FrozenBatchNorm2d`, `LayerNorm`, `GroupNorm`, `InstanceNorm2d`, `RMSNorm` |
| Activation | `ReLU`, `SiLU`, `GELU`, `Sigmoid`, `Tanh`, `PReLU`, and many more in `mu.nn.functional` |
| Pooling / shape | `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`, `AdaptiveMaxPool2d`, `Upsample`, `Flatten`, `Unflatten` |
| Attention / sequence | `MultiheadAttention`, `TransformerEncoder(Layer)`, `TransformerDecoder(Layer)`, `RotaryEmbedding`, `RNN`, `LSTM`, `GRU` |
| Regularization | `Dropout` |
| Containers | `Sequential`, `ModuleList` |

`mu.nn.functional` mirrors `torch.nn.functional`, including:

- `scaled_dot_product_attention` (fused flash kernel on the GPU);
- `interpolate`, `grid_sample`, `pad`;
- `roi_align`, `nms`, `batched_nms`.

Tensors have the usual methods: arithmetic, `@`, `reshape`, `permute`, `sum`, `mean`, `amax`, `softmax`, and
indexing. `mu.cat`, `mu.stack` and `mu.where` work as in PyTorch.

## Losses

| Task | Loss |
|---|---|
| Classification | `cross_entropy(logits, labels)` (integer labels or class probabilities; `label_smoothing`, `weight`, `ignore_index`), `nll_loss` |
| Binary / multi-label | `binary_cross_entropy_with_logits`, `binary_cross_entropy` |
| Regression | `mse_loss`, `l1_loss`, `smooth_l1_loss`, `huber_loss` |
| Distillation | `kl_div` |

They are all in `mu.nn.functional`, with module forms `mu.nn.CrossEntropyLoss` etc. Every loss defaults to
`reduction="mean"`. Any differentiable expression works as a loss: `loss = (pred - target).abs().mean() + 0.1 * reg`.

## Optimizers and learning-rate schedules

```python
optimizer = mu.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
optimizer = mu.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)

# parameter groups: e.g. no weight decay on norms and biases
decay = [p for n, p in model.named_parameters() if p.ndim > 1]
no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
optimizer = mu.optim.AdamW([{"params": decay}, {"params": no_decay, "weight_decay": 0.0}], lr=1e-3)
```

Learning rates and other group settings are **live**. Change them between calls and the next step uses them, with no
recompile:

```python
scheduler = mu.optim.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
for epoch in range(90):
    for x, y in loader:
        train_step(x, y)
    scheduler.step()

# any schedule: set it yourself (e.g. warmup + cosine, per step)
for step, (x, y) in enumerate(loader):
    lr = base_lr * min(1, step / warmup) * 0.5 * (1 + math.cos(math.pi * step / total))
    for g in optimizer.param_groups:
        g["lr"] = lr
    train_step(x, y)
```

## Evaluation

Compile a second function for inference. Switching `model.eval()` / `model.train()` is all that is needed, because
compiled functions re-trace automatically when the mode changes. In eval mode BatchNorm uses running statistics and
Dropout is off.

```python
@mu.compile
def predict(images):
    return model(images)

model.eval()
correct = 0
for images, labels in DataLoader(ArrayDataset(val_images, val_labels), batch_size=250):
    correct += (predict(images).numpy().argmax(1) == labels).sum()
model.train()
```

Both functions share the same parameters, so `predict` always sees the latest weights.

Compiled functions have **static shapes**: a different batch size needs its own compiled function. Use a validation
batch size that divides the set, or `drop_last=False` with a second function for the last batch.

## Checkpoints and resuming

```python
from munet.checkpoint import save_state, load_state

save_state({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch}, "checkpoint.mstate")

state = load_state("checkpoint.mstate")
model.load_state_dict(state["model"])
optimizer.load_state_dict(state["optimizer"])
scheduler.load_state_dict(state["scheduler"])
loader.set_epoch(state["epoch"] + 1)          # same shuffle order as an uninterrupted run
```

- **Atomic writes.** `save_state` writes atomically (temporary file, then rename), so a crash never leaves a broken
  checkpoint.
- **What it stores.** It takes nested dicts/lists of float32 arrays and plain Python values.
- **Portable.** Checkpoints work on any GPU: resume on a different card or vendor.

## Precision

`mu.compile(..., precision=...)`:

| Precision | What it does | Use for |
|---|---|---|
| `"mixed"` (default) | matmuls and convolutions use fp16 inputs with fp32 accumulation on tensor/matrix cores; everything else fp32; convolution gradients are rescaled so small values don't underflow | normal training |
| `"mixed_safe"` | `"mixed"`, plus rescaling of matmul gradients too | models with very small gradients, or training that stalls or diverges in `"mixed"` |
| `"fp32"` | full float32 | debugging, reference results |

No loss scaling code is needed. GPUs without cooperative-matrix support run fp32 in every mode.

## Gradient clipping, EMA, freezing layers

```python
ema = mu.optim.ModelEMA(model, decay=0.9999, warmups=2000)

@mu.compile
def train_step(x, y):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    mu.optim.clip_grad_norm_(model.parameters(), max_norm=1.0)   # between backward and step
    optimizer.step()
    ema.update()                                                 # runs on the GPU inside the step
    return loss

ema.copy_to(model)                                # evaluate / export the averaged weights
model.backbone.requires_grad_(False)              # freeze (before compiling the step)
```

## Use PyTorch weights

Module names and shapes follow PyTorch, so state dicts load directly:

```python
model.load_state_dict(torch_model.state_dict())          # torch tensors or NumPy arrays
model.load_state_dict(state, strict=False)               # returns missing / unexpected keys
```

To run a PyTorch model without rewriting it, import it through ONNX instead (`from_torch`, see [onnx.md](onnx.md)).
That works for inference, not training.

## Export the trained model

```python
model.eval()
program = mu.compile(model)
program(example_batch)                     # fixes the input shape and compiles
mu.save(program, "model.mnet")             # Python, C, C++, Rust: munet.load / munet_load / Program::load

from munet.interop import to_onnx
to_onnx(program, "model.onnx")             # standard ONNX for other runtimes
```

Export one program per batch size you serve. See [inference.md](inference.md) for making it fast.

## Rules of compiled functions

| Rule | Why | What to do |
|---|---|---|
| Shapes are fixed by the first call | the program is specialized for them | one compiled function per shape; `drop_last=True` in the loader |
| No Python `if` on tensor values | tracing runs Python once | use `mu.where(cond, a, b)` |
| Python side effects run only during tracing | later calls replay the GPU program | keep logging outside, return values instead |
| One `optimizer.step()` per function | one update per program | combine parameter groups in one optimizer |
| A `Result` is valid until the next call | outputs reuse their storage | `.numpy()` what you keep |
| Changing the model's structure needs a new compiled function | the program is built once | create the function after building the model |

Several GPUs: see [multi-gpu.md](multi-gpu.md). Data pipelines: see [data.md](data.md).
