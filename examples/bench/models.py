"""Bench suite manifest: ONNX models from usls (https://github.com/jamjamjon/usls, MIT; assets at
https://github.com/jamjamjon/assets/releases), Hugging Face (url=), local exports (export=, see export.py) and local
files (local=, skipped when absent). Each entry names its source and only what cannot be read from the graph: sizes for symbolic dimensions, demo images, and non-image inputs. Inputs are otherwise derived from the ONNX file:
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
    batch: int = 1                        # size of symbolic batch dimensions
    url: str = ""                         # direct download (e.g. Hugging Face) instead of a usls release asset
    export: str = ""                      # exported locally by export.py (torchvision, U-Net, Real-ESRGAN)
    local: str = ""                       # local file (not downloaded); skipped when absent
    local_env: str = ""                   # environment variable overriding `local`

    @property
    def source(self):
        return self.local or self.url or (f"export:{self.export}" if self.export else f"{self.release}/{self.file}")


def detr_sizes(n=640, batch=1):
    return {"orig_target_sizes": Input(shape=(batch, 2), kind="const", value=n, dtype="int64")}


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
    # --- batched detection ---
    Model("yolo11m-det-b8", "yolo", "v11-m-det.onnx", "standard", "detect", batch=8),
    Model("rtdetr-v1-r18-b4", "rtdetr", "v1-r18.onnx", "standard", "detect", inputs=detr_sizes(batch=4), compare="detr", batch=4),
    Model("dfine-n-b4", "d-fine", "n-coco.onnx", "standard", "detect", inputs=detr_sizes(batch=4), compare="detr", batch=4),
    # a local RT-DETR-style detector export (1280x1280 input, symbolic batch); skipped when the file is absent
    Model("local-detr-1280-b3", "", "detr-1280.onnx", "standard", "detect", size=1280, batch=3, compare="detr",
          inputs=detr_sizes(1280, 3), local="~/.cache/munet-bench/local/detr-1280.onnx", local_env="MUNET_BENCH_LOCAL_DETR"),
    Model("local-detr-1280-b6", "", "detr-1280.onnx", "full", "detect", size=1280, batch=6, compare="detr",
          inputs=detr_sizes(1280, 6), local="~/.cache/munet-bench/local/detr-1280.onnx", local_env="MUNET_BENCH_LOCAL_DETR"),
    # --- classic CNN classifiers (torchvision exports: export.py) ---
    Model("resnet50", "", "", "standard", "classify", size=224, norm=IMAGENET, image="yolo/demo-cls.jpg", export="resnet50"),
    Model("efficientnet-b0", "", "", "standard", "classify", size=224, norm=IMAGENET, image="yolo/demo-cls.jpg", export="efficientnet-b0"),
    Model("mobilenet-v3-large", "", "", "standard", "classify", size=224, norm=IMAGENET, image="yolo/demo-cls.jpg", export="mobilenet-v3-large"),
    # --- segmentation ---
    Model("segformer-b0-ade", "", "", "standard", "segment", size=512, norm=IMAGENET,
          url="https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model.onnx"),
    Model("deeplabv3-r50", "", "", "standard", "segment", size=512, norm=IMAGENET, export="deeplabv3-r50"),
    Model("unet", "", "", "standard", "segment", size=256, export="unet"),
    Model("linknet-r18", "linknet", "felixdittrich92-r18.onnx", "standard", "ocr-det", size=1024, image="db/demo-paper.png"),
    Model("birefnet-swin-t", "birefnet", "general-bb-swin-v1-tiny-epoch-232.onnx", "full", "matting", size=1024, norm=IMAGENET,
          image="modnet/demo.png"),
    # --- vision transformers ---
    Model("swin-t", "", "", "standard", "classify", size=224, norm=IMAGENET, image="yolo/demo-cls.jpg", export="swin-t"),
    Model("vit-b16", "", "", "standard", "classify", size=224, norm=IMAGENET, image="yolo/demo-cls.jpg", export="vit-b16"),
    Model("deit-s", "deit", "s-distill.onnx", "standard", "classify", size=224, norm=IMAGENET, image="yolo/demo-cls.jpg"),
    # --- text and audio encoders ---
    Model("distilbert", "", "", "standard", "embed", url="https://huggingface.co/Xenova/distilbert-base-uncased/resolve/main/onnx/model.onnx",
          inputs={"input_ids": Input(shape=(1, 128), kind="int", low=1000, high=20000, dtype="int64"),
                  "attention_mask": Input(shape=(1, 128), kind="const", value=1, dtype="int64")}),
    Model("bert-base", "", "", "standard", "embed", url="https://huggingface.co/Xenova/bert-base-uncased/resolve/main/onnx/model.onnx",
          inputs={"input_ids": Input(shape=(1, 128), kind="int", low=1000, high=20000, dtype="int64"),
                  "attention_mask": Input(shape=(1, 128), kind="const", value=1, dtype="int64"),
                  "token_type_ids": Input(shape=(1, 128), kind="const", value=0, dtype="int64")}),
    Model("whisper-tiny-encoder", "", "", "standard", "speech", url="https://huggingface.co/onnx-community/whisper-tiny/resolve/main/onnx/encoder_model.onnx",
          inputs={"input_features": Input(shape=(1, 80, 3000), kind="normal")}),
    # --- generative / restoration ---
    Model("realesrgan-x4", "", "", "standard", "super-res", size=128, export="realesrgan-x4"),
    Model("swin2sr-lightweight-x2", "swin2sr", "lightweight-x2-64.onnx", "standard", "super-res", size=256),
    # (the published SD VAE ONNX files are fp16 / ORT contrib-op graphs: exported from the diffusers layout instead)
    Model("sd-vae-decoder", "", "", "standard", "generate", export="sd-vae-decoder",
          inputs={"images": Input(shape=(1, 4, 64, 64), kind="normal")}),
    Model("lama", "", "", "standard", "inpaint", size=512, url="https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"),
]

TIERS = {"quick": 0, "standard": 1, "full": 2}


def select(tier="quick", names=None):
    """Models of `tier` and below, or those whose name contains any of `names`."""
    if names:  # exact names, else substrings
        return [m for m in MODELS if m.name in names or not any(x.name in names for x in MODELS) and any(n in m.name for n in names)]
    return [m for m in MODELS if TIERS[m.tier] <= TIERS[tier]]
