# Tested hardware

Physical GPUs MuNet has been validated on. Each one ran the GPU test suite (about 1,300 tests on one device) and the
quick bench tier, which compares 12 ONNX models against ONNX Runtime fp32 and times them. For what MuNet detects on
your own GPU, see [gpus.md](gpus.md) (`device_info()`).

Last updated 2026-11-06 (MuNet 0.8.0, plus the 0.8.1 Turing attention fix).

## Summary

| GPU                           | Architecture   | Host                                           | Driver          | Tensor-core kernels   | Result                              |
| ----------------------------- | -------------- | ---------------------------------------------- | --------------- | --------------------- | ----------------------------------- |
| NVIDIA GeForce RTX 3060       | Ampere (GA106) | x86-64, Arch Linux                             | 615.78          | yes (fp16, int8)      | **Pass**; reference development GPU |
| Intel Arc B580                | Xe2 (BMG-G21)  | x86-64, Arch Linux                             | Mesa 26.2 (ANV) | yes (fp16, int8)      | **Pass**; reference development GPU |
| NVIDIA A10G                   | Ampere (GA102) | x86-64 (AWS g5), Ubuntu 24.04                  | 595.91          | yes (fp16, int8)      | **Pass**                            |
| NVIDIA L4                     | Ada (AD104)    | x86-64 (AWS g6), Ubuntu 24.04                  | 595.91          | yes (fp16, int8)      | **Pass**                            |
| NVIDIA RTX PRO 4500 Blackwell | Blackwell      | x86-64 (AWS g7), Ubuntu 24.04                  | 595.91          | yes (fp16, int8)      | **Pass**                            |
| NVIDIA Tesla T4               | Turing (TU104) | x86-64 (AWS g4dn), Ubuntu 24.04                | 595.91          | yes (fp16, int8)      | **Pass** from 0.8.1; see the notes  |
| NVIDIA Tesla T4               | Turing (TU104) | x86-64 (AWS g4dn), Ubuntu 24.04                | 535.309         | **no**: fp32 fallback | Correct but slow; update the driver |
| NVIDIA T4G                    | Turing         | **aarch64** (AWS g5g, Graviton2), Ubuntu 24.04 | 595.91          | yes (fp16, int8)      | **Pass** from 0.8.1                 |

The CPU reference executor (`device="cpu"`) passes its suite on x86-64 and aarch64.

"Pass" means the whole GPU suite passed and every bench model matched ONNX Runtime at fp32 and fp16. A few models are
graded WARN rather than PASS (listed below); WARN still means correct output, with a precision metric just over its
threshold.

## Notes per GPU

**All NVIDIA.** Use a driver that exposes `VK_KHR_cooperative_matrix`; on Linux the 595 branch was used for every
NVIDIA card here. Containers need `NVIDIA_DRIVER_CAPABILITIES` to include `graphics`, which supplies the Vulkan driver.

**Tesla T4 with the 535 driver** (the long-term branch on many cloud images). The driver exposes tensor cores only
through `VK_NV_cooperative_matrix`, but MuNet's fp16 and int8 kernels need the KHR extension. Without it, every
`"mixed"` / `"fp16"` / int8 layer runs as fp32: results are correct, but about 2–4× slower than with tensor cores
(e.g. rtdetr-v1-r18 26.0 ms vs 7.1 ms; Depth Anything v2-s 63.7 ms vs 16.9 ms). From 0.8.0 MuNet prints a warning
for this case, and `device_info()["driver_warning"]` explains it. Updating to the 595 branch fixes it. With 535 the
suite passes apart from one test that wrongly assumed int8 support; that test was fixed in 0.8.1.

**Turing (T4, T4G): flash attention with head size above 64.** Up to 0.8.0, attention with head sizes above 64
(e.g. 96 or 128, common in LLMs) gave wrong outputs on Turing. The cause was a driver issue with a small shared-memory
array. 0.8.1 works around it. Head sizes up to 64 (most vision transformers and DETRs) were never affected, and
Ampere and newer were never affected.

**T4G (aarch64).** Same results as the x86 T4. Native int8 calibration scales can differ from Python's in the last
floating-point bit, because the compiler fuses multiply-adds on aarch64; the parity test allows that from 0.8.1.

**Intel Arc B580.** It uses 8×16 cooperative-matrix fragments on 16-lane subgroups; MuNet picks these automatically.
H.264 decode with Vulkan video (the zero-copy frame path) needs `ANV_DEBUG=video-decode` with current Mesa. See
[gpus.md](gpus.md#intel-arc-xe).

## Bench WARNs (correct output, precision just over the threshold)

| GPU                     | Model / precision                | Detail                                                                   |
| ----------------------- | -------------------------------- | ------------------------------------------------------------------------ |
| A10G, L4, T4 (595), T4G | ppocr-v4-det fp16                | max relative error ~0.11 on a few mask pixels; cosine similarity 0.99999 |
| RTX PRO 4500 Blackwell  | modnet fp16                      | max relative error 0.12 on a few matte pixels; cosine similarity 0.99999 |
| RTX 3060, Arc B580      | dinov2-s int8, ppocr-v4-det int8 | int8 quantization error (int8 runs on the reference GPUs only)           |

All of these models pass at fp32.

## Speed

These numbers come from the quick bench tier at fp16 and batch 1, as the minimum GPU time per inference:
- **MuNet:** GPU timestamps.
- **TensorRT:** 10.16 (container `nvcr.io/nvidia/tensorrt:26.03-py3`), measured with `trtexec --noDataTransfers` on the
  same GPU and at the same precision.

The RTX 3060 is the baseline for the relative table. The Arc B580 has no TensorRT. With the 535 driver (fp32 fallback),
the T4 runs at **0.27×** the RTX 3060 (geometric mean), against 0.64× with the 595 driver.

### MuNet fp16: GPU ms per inference (batch 1, lower is better)

| Model               | RTX 3060 | Arc B580 | T4    | T4G   | A10G | L4   | RTX PRO 4500 Blackwell |
| ------------------- | -------- | -------- | ----- | ----- | ---- | ---- | ---------------------- |
| convnext-v2-atto    | 0.77     | 0.95     | 1.15  | 1.23  | 0.76 | 0.61 | 0.60                   |
| depth-anything-v2-s | 10.64    | 12.49    | 17.00 | 19.99 | 5.55 | 5.99 | 3.44                   |
| dfine-n             | 2.40     | 3.72     | 3.75  | 4.09  | 2.07 | 1.67 | 1.53                   |
| dinov2-s            | 1.77     | 2.43     | 2.48  | 2.96  | 1.54 | 1.21 | 1.06                   |
| fastvit-t8          | 0.69     | 1.03     | 1.08  | 1.19  | 0.65 | 0.48 | 0.41                   |
| mobile-sam-encoder  | 11.53    | 9.62     | 17.25 | 18.94 | 6.41 | 6.05 | 3.73                   |
| mobileone-s0        | 0.28     | 0.60     | 0.50  | 0.54  | 0.33 | 0.26 | 0.24                   |
| modnet              | 6.19     | 5.05     | 8.76  | 9.70  | 3.82 | 3.50 | 2.29                   |
| ppocr-v4-det        | 1.82     | 1.84     | 2.39  | 2.52  | 1.08 | 0.93 | 0.68                   |
| ppocr-v4-rec-en     | 0.65     | 0.76     | 1.14  | 1.20  | 0.68 | 0.50 | 0.51                   |
| rtdetr-v1-r18       | 4.53     | 4.69     | 7.12  | 9.66  | 3.57 | 2.99 | 2.36                   |
| yolo11n-det         | 1.32     | 1.78     | 2.34  | 2.69  | 1.17 | 0.89 | 0.73                   |
| **geometric mean**  | 1.97     | 2.41     | 3.06  | 3.44  | 1.56 | 1.31 | 1.05                   |

### Speed relative to RTX 3060 (higher is faster; RTX 3060 = 1.00×)

| Model               | RTX 3060 | Arc B580 | T4    | T4G   | A10G  | L4    | RTX PRO 4500 Blackwell |
| ------------------- | -------- | -------- | ----- | ----- | ----- | ----- | ---------------------- |
| convnext-v2-atto    | 1.00×    | 0.81×    | 0.67× | 0.63× | 1.01× | 1.25× | 1.27×                  |
| depth-anything-v2-s | 1.00×    | 0.85×    | 0.63× | 0.53× | 1.92× | 1.78× | 3.09×                  |
| dfine-n             | 1.00×    | 0.64×    | 0.64× | 0.59× | 1.16× | 1.43× | 1.57×                  |
| dinov2-s            | 1.00×    | 0.73×    | 0.72× | 0.60× | 1.15× | 1.46× | 1.68×                  |
| fastvit-t8          | 1.00×    | 0.67×    | 0.64× | 0.58× | 1.07× | 1.43× | 1.69×                  |
| mobile-sam-encoder  | 1.00×    | 1.20×    | 0.67× | 0.61× | 1.80× | 1.91× | 3.09×                  |
| mobileone-s0        | 1.00×    | 0.47×    | 0.56× | 0.51× | 0.86× | 1.08× | 1.14×                  |
| modnet              | 1.00×    | 1.23×    | 0.71× | 0.64× | 1.62× | 1.77× | 2.70×                  |
| ppocr-v4-det        | 1.00×    | 0.99×    | 0.76× | 0.72× | 1.68× | 1.96× | 2.66×                  |
| ppocr-v4-rec-en     | 1.00×    | 0.85×    | 0.57× | 0.54× | 0.95× | 1.29× | 1.27×                  |
| rtdetr-v1-r18       | 1.00×    | 0.96×    | 0.64× | 0.47× | 1.27× | 1.52× | 1.92×                  |
| yolo11n-det         | 1.00×    | 0.74×    | 0.56× | 0.49× | 1.13× | 1.48× | 1.80×                  |
| **geometric mean**  | 1.00×    | 0.82×    | 0.64× | 0.57× | 1.26× | 1.51× | 1.88×                  |

### MuNet vs TensorRT fp16 on the same GPU: MuNet ms / TensorRT ms (MuNet speed as % of TensorRT)

| Model                                     | RTX 3060           | T4                  | T4G                 | A10G                | L4                  | RTX PRO 4500 Blackwell |
| ----------------------------------------- | ------------------ | ------------------- | ------------------- | ------------------- | ------------------- | ---------------------- |
| convnext-v2-atto                          | 0.77 / 0.57 (74%)  | 1.15 / 0.76 (66%)   | 1.23 / 0.81 (65%)   | 0.76 / 0.55 (73%)   | 0.61 / 0.46 (75%)   | 0.60 / 0.40 (66%)      |
| depth-anything-v2-s                       | 10.64 / – (–)      | 17.00 / – (–)       | 19.99 / – (–)       | 5.55 / – (–)        | 5.99 / – (–)        | 3.44 / – (–)           |
| dfine-n                                   | 2.40 / 1.63 (68%)  | 3.75 / 1.96 (52%)   | 4.09 / 2.34 (57%)   | 2.07 / 1.29 (62%)   | 1.67 / 1.05 (63%)   | 1.53 / 1.09 (71%)      |
| dinov2-s                                  | 1.77 / 0.99 (56%)  | 2.48 / 1.32 (53%)   | 2.96 / 1.83 (62%)   | 1.54 / 0.79 (51%)   | 1.21 / 0.66 (54%)   | 1.06 / 0.38 (36%)      |
| fastvit-t8                                | 0.69 / 6.47 (938%) | 1.08 / 9.34 (862%)  | 1.19 / 9.37 (785%)  | 0.65 / 7.66 (1183%) | 0.48 / 6.27 (1298%) | 0.41 / 0.47 (114%)     |
| mobile-sam-encoder                        | 11.53 / 7.61 (66%) | 17.25 / 10.98 (64%) | 18.94 / 14.98 (79%) | 6.41 / 4.60 (72%)   | 6.05 / 4.80 (79%)   | 3.73 / 1.97 (53%)      |
| mobileone-s0                              | 0.28 / 0.19 (70%)  | 0.50 / 0.23 (47%)   | 0.54 / 0.25 (46%)   | 0.33 / 0.19 (57%)   | 0.26 / 0.16 (62%)   | 0.24 / 0.18 (74%)      |
| modnet                                    | 6.19 / 3.39 (55%)  | 8.76 / 4.90 (56%)   | 9.70 / 6.37 (66%)   | 3.82 / 2.23 (58%)   | 3.50 / 2.00 (57%)   | 2.29 / 0.98 (43%)      |
| ppocr-v4-det                              | 1.82 / 1.22 (67%)  | 2.39 / 1.68 (70%)   | 2.52 / 1.91 (76%)   | 1.08 / 0.88 (82%)   | 0.93 / 0.74 (80%)   | 0.68 / 0.45 (65%)      |
| ppocr-v4-rec-en                           | 0.65 / 0.42 (65%)  | 1.14 / 0.52 (46%)   | 1.20 / 0.59 (49%)   | 0.68 / 0.40 (58%)   | 0.50 / 0.34 (68%)   | 0.51 / 0.32 (62%)      |
| rtdetr-v1-r18                             | 4.53 / 3.27 (72%)  | 7.12 / 4.35 (61%)   | 9.66 / 6.75 (70%)   | 3.57 / 2.34 (66%)   | 2.99 / 2.11 (71%)   | 2.36 / 1.13 (48%)      |
| yolo11n-det                               | 1.32 / 1.20 (91%)  | 2.34 / 1.51 (65%)   | 2.69 / 1.86 (69%)   | 1.17 / 1.01 (86%)   | 0.89 / 0.80 (90%)   | 0.73 / 0.68 (93%)      |
| **geometric mean** (excluding fastvit-t8) | 68%                | 57%                 | 63%                 | 66%                 | 69%                 | 59%                    |

TensorRT notes:
- **depth-anything-v2-s:** TensorRT 10.16 failed to build an engine on every GPU. MuNet runs it.
- **fastvit-t8:** TensorRT picked a slow plan (6–9 ms) on every GPU before Blackwell, so it is left out of the TensorRT
  mean, as noted in the table.

The tables are generated with `examples/bench/hardware_report.py` from `examples/bench/run.py --trt` results. For
contributors, the engine repository's `tools/hwtest` runs the whole validation on a GPU host or on AWS.

## Not yet tested

- **AMD Radeon** (RDNA2/3/4) and AMD Instinct. AWS no longer offers its only AMD GPU instance (g4ad). An RDNA4 card
  is planned for local testing. RDNA3 and newer expose cooperative matrices through Mesa RADV; RDNA2 and older run fp32
  kernels.
- **Apple Silicon** through MoltenVK (no cooperative matrices expected, so fp32 kernels).
- **Intel integrated graphics and Arc A-series (Alchemist).**
- **Mobile GPUs** (Arm Mali, Qualcomm Adreno).
- **NVIDIA data-centre GPUs** (A100, H100, B200) and Windows.

Reports from other hardware are welcome. Include the output of `munet.device_info()` and the bench results
(`python examples/bench/run.py --tier quick`).
