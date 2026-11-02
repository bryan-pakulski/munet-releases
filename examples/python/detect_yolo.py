"""Object detection with a YOLO ONNX model (YOLOv8 / YOLO11 exports: output [1, 4 + classes, anchors]).

    curl -LO https://github.com/jamjamjon/assets/releases/download/yolo/v11-n-det.onnx
    python examples/python/detect_yolo.py v11-n-det.onnx street.jpg                 # writes street.det.jpg
    python examples/python/detect_yolo.py v11-n-det.onnx street.jpg --precision fp16 --save yolo.mnet

Pre-processing (letterbox to 640x640, RGB in [0, 1]) and post-processing (score filter, class-wise NMS, undo the
letterbox) are plain NumPy; the network runs on the GPU. Ultralytics exports work too:
`yolo export model=yolo11n.pt format=onnx`. Needs Pillow (`pip install pillow`).
"""
import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import munet as mu
from munet.interop import from_onnx

COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic-light fire-hydrant stop-sign parking-meter "
        "bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
        "skis snowboard sports-ball kite baseball-bat baseball-glove skateboard surfboard tennis-racket bottle "
        "wine-glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hot-dog pizza donut cake "
        "chair couch potted-plant bed dining-table toilet tv laptop mouse remote keyboard cell-phone microwave oven "
        "toaster sink refrigerator book clock vase scissors teddy-bear hair-drier toothbrush").split()


def letterbox(image, size):
    """Resize keeping the aspect ratio and pad to size x size (grey 114, like Ultralytics). Returns NCHW float32."""
    scale = min(size / image.width, size / image.height)
    w, h = round(image.width * scale), round(image.height * scale)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    left, top = (size - w) // 2, (size - h) // 2
    canvas.paste(image.convert("RGB").resize((w, h), Image.BILINEAR), (left, top))
    x = np.asarray(canvas, np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(x), scale, left, top


def nms(boxes, scores, iou):
    """Greedy NMS on xyxy boxes; returns kept indices, best first."""
    order, keep = scores.argsort()[::-1], []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        xx1, yy1 = np.maximum(boxes[i, 0], boxes[rest, 0]), np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2, yy2 = np.minimum(boxes[i, 2], boxes[rest, 2]), np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        order = rest[inter / (area[i] + area[rest] - inter + 1e-9) <= iou]
    return np.array(keep, int)


def decode(output, scale, left, top, conf=0.25, iou=0.45):
    """[4 + classes, anchors] (cx, cy, w, h in input pixels, then class scores) -> (xyxy in image pixels, score, class)."""
    pred = output.T
    cls = pred[:, 4:].argmax(1)
    score = pred[np.arange(len(pred)), 4 + cls]
    pred, cls, score = pred[score > conf], cls[score > conf], score[score > conf]
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    keep = nms(boxes + cls[:, None] * 4096.0, score, iou)  # offset per class: class-wise NMS in one pass
    boxes = (boxes[keep] - [left, top, left, top]) / scale
    return boxes, score[keep], cls[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("image")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--precision", default="mixed", choices=["fp32", "mixed", "fp16"])
    ap.add_argument("--device", default="vulkan")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save", help="also write the compiled program as .mnet (for C, C++, Rust)")
    args = ap.parse_args()

    image = Image.open(args.image)
    x, scale, left, top = letterbox(image, args.size)

    # Static input shape: one compiled program per resolution and batch size. The first call compiles
    # (seconds; kernels are cached on disk for the next run), later calls only run.
    model = from_onnx(args.model, device=args.device, precision=args.precision,
                      input_shapes={"images": (1, 3, args.size, args.size)})
    out = model(x)
    t = time.perf_counter()
    for _ in range(20):
        out = model(x)
    raw = out.numpy()[0]  # [84, 8400]; reading the result waits for the GPU
    print(f"{(time.perf_counter() - t) / 20 * 1e3:.2f} ms per image (upload, network, readback); "
          f"GPU {model.stats()['gpu_ms']:.2f} ms")

    boxes, scores, classes = decode(raw, scale, left, top, conf=args.conf)
    draw = ImageDraw.Draw(image := image.convert("RGB"))
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        name = COCO[c] if c < len(COCO) else str(c)
        print(f"{name:14s} {s:.2f}  [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 64), width=3)
        draw.text((x1 + 4, y1 + 2), f"{name} {s:.2f}", fill=(255, 64, 64))
    target = Path(args.image).with_suffix(".det.jpg")
    image.save(target)
    print(f"{len(boxes)} detections -> {target}")
    if args.save:
        mu.save(model, args.save)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
