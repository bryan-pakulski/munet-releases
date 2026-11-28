# Hardware report

`munet.diagnose` collects what we need to tune MuNet for a GPU we don't have. Run it, then attach the file it writes
to a [hardware report issue](https://github.com/bryan-pakulski/munet-releases/issues/new?template=hardware-report.yml).

## Run it

1. Install MuNet, a Vulkan driver and `glslangValidator` ([getting-started.md](getting-started.md)).
2. Close games, video and other GPU work. Plug in laptops.
3. Run:
   ```bash
   python -m munet.diagnose            # every GPU, 10-15 minutes each
   ```
4. Open a [hardware report](https://github.com/bryan-pakulski/munet-releases/issues/new?template=hardware-report.yml)
   and attach `munet-diag-<gpu>.json.gz`.

If MuNet finds no GPU, run it anyway: the report records the Vulkan error.

**Android phone:** `python tools/phone_diagnose.py` from the unpacked
[`munet-android-<version>`](android.md#hardware-report-for-your-phone) archive, over `adb` (5 minutes, `--quick` 2). It
records the same sections as below, measured through the C API on the phone, plus the phone's model, SoC, memory and
thermal state, and writes `munet-diag-phone-<model>.json.gz`. Attach that file the same way. It has no tile sweep.

| Option | Effect |
|---|---|
| `--quick` | shorter run, 2-3 minutes per GPU |
| `--device N` | only Vulkan device N (repeatable) |
| `--budget S` | seconds for the tile sweep per GPU (default 600, `--quick` 60) |
| `--out FILE` | output path |

## What it collects

- **Device:** driver name and version, Vulkan features, limits, extensions and cooperative-matrix shapes.
- **Ceilings:** measured memory bandwidth and fp32 / fp16 / int8 throughput.
- **Correctness:** GEMM, convolution, attention, normalization and a training step at fp32, mixed and fp16,
  compared with MuNet's CPU backend.
- **Tile sweep:** the time of every candidate tile for common GEMM and convolution shapes, for inference and for
  training (input and weight gradients). This shows which defaults are wrong for your GPU.
- **Model profiles:** per-kernel GPU times of a small CNN and a small ViT.

## Privacy

- No model downloads and no network access.
- No host name, user name, file paths or PCI slot. Error messages and environment values are scrubbed of them.
- Only `MUNET_*`, `VK_*`, `RADV_*`, `ANV_*`, `MESA_*`, `AMD_*` and `NV_*` environment variables are recorded.
- The file is plain JSON. Read it before you share it: `zcat munet-diag-*.json.gz | less`.
