"""Bench suite manifest: ONNX models from usls (https://github.com/jamjamjon/usls, MIT; assets at
https://github.com/jamjamjon/assets/releases). Each entry names a release asset and only what cannot be read from the
graph: sizes for symbolic dimensions, demo images, and non-image inputs. Inputs are otherwise derived from the ONNX file:
4-D float inputs with 3 channels are images (the demo image, RGB in [0, 1], optionally normalized), everything else is
seeded random data of the declared type.

Tiers: "quick" (one or two per family, small; minutes), "standard" (larger and more variants), "full" (everything
listed). Add models freely: `python examples/bench/run.py --models NAME` downloads and pins them on first use.
"""
from dataclasses import dataclass, field

BASE = "https://github.com/jamjamjon/assets/releases/download"

IMAGENET = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
CLIP_NORM = dict(mean=(0.4815, 0.4578, 0.4082), std=(0.2686, 0.2613, 0.2758))


@dataclass
class Input:
    """Override for one model input. kind: "image" (demo image resized to shape[2:], RGB [0,1], then normalized),
    "uniform"/"normal" (random float), "int" (random integers in [low, high)), "const" (filled with value)."""
    shape: tuple = None
    kind: str = None
    image: str = "yolo/demo-det.jpg"
    mean: tuple = None
    std: tuple = None
    low: float = 0
    high: float = 1
    value: float = 0
    dtype: str = None


@dataclass
class Model:
    name: str
    release: str
    file: str
    tier: str = "standard"
    task: str = ""
    size: int = 640                       # default spatial size for symbolic H/W of image inputs
    inputs: dict = field(default_factory=dict)  # input name -> Input overrides
    norm: dict = field(default_factory=dict)    # mean/std for all image inputs
    image: str = "yolo/demo-det.jpg"
    notes: str = ""
    compare: str = ""                     # "detr": outputs (labels, boxes, scores) compared as matched detections

    @property
    def url(self):
        return f"{BASE}/{self.release}/{self.file}"


def detr_sizes(n=640):
    return {"orig_target_sizes": Input(shape=(1, 2), kind="const", value=n, dtype="int64")}


MODELS = [
    # --- detection: YOLO family (anchor-free heads, SiLU CNNs) ---
    Model("yolo11n-det", "yolo", "v11-n-det.onnx", "quick", "detect"),
    Model("yolo11m-det", "yolo", "v11-m-det.onnx", "standard", "detect"),
    Model("yolo11l-det", "yolo", "v11-l-det.onnx", "full", "detect"),
    Model("yolo11n-det-q8", "yolo", "v11-n-det-q8.onnx", "standard", "detect", notes="usls dynamic int8 ONNX"),
    Model("yolo10n-det", "yolo", "v10-n-det.onnx", "standard", "detect", notes="NMS-free head (TopK/Gather)"),
    Model("yolo11n-seg", "yolo", "v11-n-seg.onnx", "standard", "segment"),
    Model("yolo11n-pose", "yolo", "v11-n-pose.onnx", "standard", "pose", image="yolo/demo-pose.jpg"),
    Model("yolo11n-obb", "yolo", "v11-n-obb.onnx", "full", "obb", image="yolo/demo-obb.jpg"),
    Model("yolo11m-cls", "yolo", "v11-m-cls.onnx", "standard", "classify", size=224, image="yolo/demo-cls.jpg"),
    Model("yolov8n-v0", "v0.0.1", "yolov8n-dyn.onnx", "full", "detect"),
    Model("yolov5n-v0", "v0.0.1", "yolov5n-dyn.onnx", "full", "detect"),
    Model("yolov7-tiny", "v0.0.1", "yolov7-tiny-dyn.onnx", "full", "detect"),
    # --- detection: DETR family (transformer decoders, deformable attention, grid sampling) ---
    Model("rtdetr-v1-r18", "rtdetr", "v1-r18.onnx", "quick", "detect", inputs=detr_sizes(), compare="detr"),
    Model("rtdetr-v1-r101", "rtdetr", "v1-r101.onnx", "full", "detect", inputs=detr_sizes(), compare="detr"),
    Model("rtdetr-v1-r18-q8", "rtdetr", "v1-r18-q8.onnx", "full", "detect", inputs=detr_sizes(), notes="usls int8 ONNX", compare="detr"),
    Model("rtdetr-l", "yolo", "rtdetr-l-det.onnx", "standard", "detect"),
    Model("dfine-n", "d-fine", "n-coco.onnx", "quick", "detect", inputs=detr_sizes(), compare="detr"),
    Model("dfine-m", "d-fine", "m-coco.onnx", "full", "detect", inputs=detr_sizes(), compare="detr"),
    Model("deim-dfine-s", "deim", "dfine-s-coco.onnx", "standard", "detect", inputs=detr_sizes(), compare="detr"),
    Model("rfdetr-base", "rfdetr", "base.onnx", "standard", "detect", size=560, norm=IMAGENET),
    # --- classification / backbones (CNNs, re-parameterized nets, ViTs) ---
    Model("convnext-v2-atto", "convnext", "v2-a.onnx", "quick", "classify", size=224, norm=IMAGENET, image="yolo/demo-cls.jpg"),
    Model("convnext-v2-tiny", "convnext", "v2-t.onnx", "standard", "classify", size=224, norm=IMAGENET, image="yolo/demo-cls.jpg"),
    Model("convnext-v1-tiny", "convnext", "v1-t.onnx", "full", "classify", size=224, norm=IMAGENET, image="yolo/demo-cls.jpg"),
    Model("mobileone-s0", "mobileone", "s0.onnx", "quick", "classify", size=224, norm=IMAGENET, image="yolo/demo-cls.jpg"),
    Model("mobileone-s4-256", "mobileone", "s4-256x256.onnx", "full", "classify", size=256, norm=IMAGENET, image="yolo/demo-cls.jpg"),
    Model("fastvit-t8", "fastvit", "t8.onnx", "quick", "classify", size=256, norm=IMAGENET, image="yolo/demo-cls.jpg"),
    Model("fastvit-sa12", "fastvit", "sa12.onnx", "standard", "classify", size=256, norm=IMAGENET, image="yolo/demo-cls.jpg"),
    Model("dinov2-s", "dinov2", "s.onnx", "quick", "embed", size=224, norm=IMAGENET, image="yolo/demo-cls.jpg"),
    Model("dinov2-b", "dinov2", "b.onnx", "full", "embed", size=224, norm=IMAGENET, image="yolo/demo-cls.jpg"),
    Model("clip-b32-visual", "v0.0.1", "clip-b32-visual-dyn.onnx", "standard", "embed", size=224, norm=CLIP_NORM, image="yolo/demo-cls.jpg"),
    # --- dense prediction: depth, matting, segmentation encoders ---
    Model("depth-anything-v2-s", "v0.0.1", "depth-anything-v2-s.onnx", "quick", "depth", size=518, norm=IMAGENET),
    Model("modnet", "modnet", "photographic-portrait-matting.onnx", "quick", "matting", size=512,
          norm=dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), image="modnet/demo.png"),
    Model("mobile-sam-encoder", "sam", "mobile-sam-vit-t-encoder.onnx", "quick", "segment", size=1024, image="sam/demo-dog.png",
          norm=dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))),
    Model("sam2-tiny-encoder", "sam", "sam2-hiera-tiny-encoder.onnx", "full", "segment", size=1024, image="sam/demo-dog.png", norm=IMAGENET),
    Model("fastsam-s", "yolo", "FastSAM-s.onnx", "standard", "segment"),
    # --- OCR (thin, wide inputs; LSTM-free CTC recognizers) ---
    Model("ppocr-v4-det", "db", "ppocr-v4-ch.onnx", "quick", "ocr-det", size=640, norm=IMAGENET, image="db/demo-paper.png"),
    Model("ppocr-v4-rec-en", "svtr", "ppocr-v4-en.onnx", "quick", "ocr-rec", image="svtr/1.png",
          inputs={"x": Input(shape=(1, 3, 48, 320), kind="image", image="svtr/1.png",
                             mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))}),
    Model("picodet-layout", "picodet", "layout-1x.onnx", "standard", "detect", size=800, image="picodet/demo-layout-1x.png"),
]

TIERS = {"quick": 0, "standard": 1, "full": 2}


def select(tier="quick", names=None):
    """Models of `tier` and below, or those whose name contains any of `names`."""
    if names:  # exact names, else substrings
        return [m for m in MODELS if m.name in names or not any(x.name in names for x in MODELS) and any(n in m.name for n in names)]
    return [m for m in MODELS if TIERS[m.tier] <= TIERS[tier]]
