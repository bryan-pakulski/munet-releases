"""Train an image classifier end to end: data loader, compiled training step, evaluation, checkpoints, export.

    python examples/python/train_classifier.py                      # trains on synthetic data, ~1 minute
    python examples/python/train_classifier.py --resume             # continue from the last checkpoint
    python examples/python/train_classifier.py --device cpu --epochs 1

Outputs in --out (default runs/classifier): checkpoint.mstate (resume), classifier.mnet (load from Python, C, C++ or
Rust) and classifier.onnx. The data is synthetic (a bright square in one of four quadrants, plus noise) so the script
runs anywhere; replace `make_dataset` with your own arrays or Dataset.
"""
import argparse
import time
from pathlib import Path

import numpy as np

import munet as mu
from munet.checkpoint import load_state, save_state
from munet.data import ArrayDataset, DataLoader
from munet.interop import to_onnx

CLASSES = 4


def make_dataset(n, seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, CLASSES, n)
    images = rng.normal(0, 0.5, (n, 3, 32, 32)).astype(np.float32)
    for i, c in enumerate(labels):
        y, x = (c // 2) * 16 + rng.integers(2, 6), (c % 2) * 16 + rng.integers(2, 6)
        images[i, :, y:y + 8, x:x + 8] += 2.0
    return images, labels


def augment(sample):
    image, label = sample
    if np.random.random() < 0.5:  # brightness jitter; keep augmentations label-preserving
        image = image * np.float32(np.random.uniform(0.8, 1.2))
    return image, label


class Net(mu.nn.Module):
    def __init__(self, rng):
        def block(cin, cout):
            return [mu.nn.Conv2d(cin, cout, 3, padding=1, rng=rng), mu.nn.BatchNorm2d(cout), mu.nn.ReLU()]
        self.features = mu.nn.Sequential(*block(3, 32), mu.nn.MaxPool2d(2), *block(32, 64), mu.nn.MaxPool2d(2),
                                         *block(64, 128), mu.nn.AdaptiveAvgPool2d(1), mu.nn.Flatten())
        self.head = mu.nn.Linear(128, CLASSES, rng=rng)

    def forward(self, x):
        return self.head(self.features(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="vulkan")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="runs/classifier")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_x, train_y = make_dataset(4096, seed=0)
    val_x, val_y = make_dataset(1024, seed=1)
    train = DataLoader(ArrayDataset(train_x, train_y, transform=augment), batch_size=args.batch, shuffle=True, workers=4)
    val = DataLoader(ArrayDataset(val_x, val_y), batch_size=256)  # 1024 / 256: no partial batch to drop

    model = Net(np.random.default_rng(0))
    optimizer = mu.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = mu.optim.MultiStepLR(optimizer, milestones=[3, 4], gamma=0.1)  # stepped once per epoch

    # The whole step (forward, loss, backward, optimizer) compiles to one GPU program on the first call;
    # later calls replay it. Parameters and optimizer state stay on the GPU between calls.
    @mu.compile(device=args.device)
    def train_step(images, labels):
        optimizer.zero_grad()
        loss = mu.nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        return loss

    @mu.compile(device=args.device)
    def predict(images):
        return model(images)  # traced in eval mode: BatchNorm uses its running statistics

    start_epoch = 0
    checkpoint = out / "checkpoint.mstate"
    if args.resume and checkpoint.exists():
        state = load_state(checkpoint)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        print(f"resumed from epoch {state['epoch']}")

    for epoch in range(start_epoch, args.epochs):
        train.set_epoch(epoch)  # shuffle order depends only on (seed, epoch): resumable and reproducible
        model.train()
        t = time.perf_counter()
        for images, labels in train:
            loss = train_step(images, labels)
        # .item() / .numpy() read back from the GPU and wait for it, so avoid them every step. A Result is only valid
        # until the next call of the same program: read the ones you need before calling again.
        last_loss = loss.item()
        scheduler.step()

        model.eval()
        correct = 0
        for images, labels in val:
            correct += int((predict(images).numpy().argmax(1) == labels).sum())
        print(f"epoch {epoch}: loss {last_loss:.4f}  val acc {correct / len(val_y):.3f}  "
              f"lr {optimizer.param_groups[0]['lr']:.1e}  {time.perf_counter() - t:.1f} s")
        save_state({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch}, checkpoint)

    # Export an inference program: batch 1 for latency (export one program per batch size you serve).
    model.eval()
    inference = mu.compile(model, device=args.device)
    inference(val_x[:1])  # compile (shapes are fixed by the first call)
    mu.save(inference, out / "classifier.mnet")
    to_onnx(inference, out / "classifier.onnx")
    print(f"saved {out / 'classifier.mnet'} and {out / 'classifier.onnx'}")


if __name__ == "__main__":
    main()
