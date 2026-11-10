"""Exports the bench models that have no usable ONNX file (torchvision models, a U-Net, Real-ESRGAN's RRDBNet, Stable
Diffusion's VAE decoder).

    python examples/bench/export.py NAME OUT.onnx

run.py calls this on first use (with $MUNET_BENCH_TORCH_PYTHON, default: its own interpreter; needs torch and
torchvision). Pretrained weights come from torchvision's download server; Real-ESRGAN's from Hugging Face, loaded with
torch.load(weights_only=True). Every model has a dynamic batch dimension.
"""
import sys
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def classifier(fn, size):
    return getattr(torchvision.models, fn)(weights="DEFAULT").eval(), (1, 3, size, size)


class SegOut(nn.Module):
    """torchvision segmentation models return a dict; export its "out" tensor."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)["out"]


class UNet(nn.Module):
    """The original U-Net layout (Ronneberger et al. 2015) with padded 3x3 convs, BatchNorm and bilinear upsampling."""
    def __init__(self, ci=3, classes=2, w=64):
        super().__init__()
        block = lambda a, b: nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                           nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        chans = [w, 2 * w, 4 * w, 8 * w, 16 * w]
        self.down = nn.ModuleList([block(ci, chans[0])] + [block(chans[i], chans[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([block(chans[i + 1] + chans[i], chans[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(w, classes, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for u, s in zip(self.up, reversed(skips[:-1])):
            x = u(torch.cat([F.interpolate(x, size=s.shape[2:], mode="bilinear", align_corners=False), s], 1))
        return self.head(x)


class RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(nf, gc, 3, 1, 1), nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3, self.conv4 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1), nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)

    def forward(self, x):
        a = lambda t: F.leaky_relu(t, 0.2)
        x1 = a(self.conv1(x))
        x2 = a(self.conv2(torch.cat((x, x1), 1)))
        x3 = a(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = a(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = RDB(nf, gc), RDB(nf, gc), RDB(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRGAN x4 generator (basicsr RRDBNet: 23 RRDB blocks, nearest upsampling x2 twice)."""
    def __init__(self, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1, self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1), nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr, self.conv_last = nn.Conv2d(nf, nf, 3, 1, 1), nn.Conv2d(nf, 3, 3, 1, 1)

    def forward(self, x):
        a = lambda t: F.leaky_relu(t, 0.2)
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = a(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = a(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(a(self.conv_hr(feat)))


def realesrgan():
    m = RRDBNet().eval()
    weights = Path(sys.argv[2]).with_suffix(".pth")
    if not weights.exists():
        urllib.request.urlretrieve("https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth", weights)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = state.get("params_ema", state.get("params", state))
    m.load_state_dict({k.replace("RDB", "rdb").replace("trunk_conv", "conv_body").replace("upconv", "conv_up")
                       .replace("HRconv", "conv_hr").replace("RRDB_trunk", "body"): v for k, v in state.items()})
    return m, (1, 3, 128, 128)


def unet():
    torch.manual_seed(0)
    m = UNet().eval()
    with torch.no_grad():  # non-trivial BatchNorm statistics (the model is randomly initialized)
        for b in m.modules():
            if isinstance(b, nn.BatchNorm2d):
                b.running_mean.uniform_(-0.1, 0.1)
                b.running_var.uniform_(0.5, 1.5)
    return m, (1, 3, 256, 256)


class VaeResnet(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(32, a, eps=1e-6), nn.Conv2d(a, b, 3, padding=1)
        self.norm2, self.conv2 = nn.GroupNorm(32, b, eps=1e-6), nn.Conv2d(b, b, 3, padding=1)
        self.skip = nn.Conv2d(a, b, 1) if a != b else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))


class VaeAttention(nn.Module):
    """Single-head self-attention over the pixels of the mid block (diffusers' VAE attention)."""
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(32, c, eps=1e-6)
        self.q, self.k, self.v, self.o = (nn.Linear(c, c) for _ in range(4))

    def forward(self, x):
        n, c, hgt, wid = x.shape
        t = self.norm(x).reshape(n, c, hgt * wid).transpose(1, 2)
        y = F.scaled_dot_product_attention(self.q(t), self.k(t), self.v(t))
        return x + self.o(y).transpose(1, 2).reshape(n, c, hgt, wid)


class VaeDecoder(nn.Module):
    """Stable Diffusion's AutoencoderKL decoder (diffusers layout: 4 latent channels, 512-512-256-128 up blocks of 3
    ResNet blocks with GroupNorm 32 + SiLU, mid-block attention, nearest 2x upsampling)."""
    def __init__(self):
        super().__init__()
        self.post_quant, self.conv_in = nn.Conv2d(4, 4, 1), nn.Conv2d(4, 512, 3, padding=1)
        self.mid = nn.ModuleList([VaeResnet(512, 512), VaeAttention(512), VaeResnet(512, 512)])
        chans = [512, 512, 256, 128]
        self.ups = nn.ModuleList()
        prev = 512
        for i, c in enumerate(chans):
            blocks = nn.ModuleList([VaeResnet(prev if j == 0 else c, c) for j in range(3)])
            self.ups.append(nn.ModuleDict({"blocks": blocks, "up": nn.Conv2d(c, c, 3, padding=1) if i < 3 else nn.Identity()}))
            prev = c
        self.norm_out, self.conv_out = nn.GroupNorm(32, 128, eps=1e-6), nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, z):
        x = self.conv_in(self.post_quant(z / 0.18215))
        for m in self.mid:
            x = m(x)
        for i, u in enumerate(self.ups):
            for blk in u["blocks"]:
                x = blk(x)
            if i < 3:
                x = u["up"](F.interpolate(x, scale_factor=2, mode="nearest"))
        return self.conv_out(F.silu(self.norm_out(x)))


def vae_decoder():
    torch.manual_seed(0)
    m = VaeDecoder().eval()
    with torch.no_grad():  # random but well-scaled weights (the decoder is randomly initialized)
        for mod in m.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(mod.weight, std=(1.0 / mod.weight[0].numel()) ** 0.5)
    return m, (1, 4, 64, 64)


BUILDERS = {
    "sd-vae-decoder": vae_decoder,
    "resnet50": lambda: classifier("resnet50", 224),
    "efficientnet-b0": lambda: classifier("efficientnet_b0", 224),
    "mobilenet-v3-large": lambda: classifier("mobilenet_v3_large", 224),
    "swin-t": lambda: classifier("swin_t", 224),
    "vit-b16": lambda: classifier("vit_b_16", 224),
    "deeplabv3-r50": lambda: (SegOut(torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT").eval()), (1, 3, 512, 512)),
    "unet": unet,
    "realesrgan-x4": realesrgan,
}


def main():
    name, out = sys.argv[1], sys.argv[2]
    torch.backends.mha.set_fastpath_enabled(False)  # the fused attention fast path has no ONNX export
    model, shape = BUILDERS[name]()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = out + ".tmp"
    with torch.no_grad():
        torch.onnx.export(model, torch.rand(*shape), tmp, input_names=["images"], output_names=["output"],
                          dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}}, opset_version=17, dynamo=False)
    Path(tmp).replace(out)


if __name__ == "__main__":
    main()
