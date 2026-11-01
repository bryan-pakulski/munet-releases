# MuNet releases

Prebuilt MuNet binaries: a neural-network runtime that compiles models (ONNX imports) into fused Vulkan compute
kernels and runs them on any Vulkan GPU — AMD, Intel, NVIDIA, Apple (MoltenVK) and mobile — with fp16/int8
cooperative-matrix kernels where the GPU has them.

| Package | Install |
|---|---|
| Python | `pip install munet-nn` |
| Rust | `cargo add munet-nn` (imported as `munet`; the build script downloads the C API below and verifies its SHA-256) |
| C | `munet-c-<version>-<os>-<arch>` archives on the [releases](https://github.com/bryan-pakulski/munet-releases/releases): `include/munet/munet_c.h`, shared and static libraries |
| Native tools | `munet-node`, `munet-server` and the C++ SDK archives on the releases |

Linux x86_64 and aarch64 (glibc 2.28+) are built for every release. Running on a GPU needs a Vulkan 1.1 driver and
`glslangValidator` (`glslang-tools` / `glslang` packages or the Vulkan SDK).

## License

The MuNet engine binaries are proprietary (see [LICENSE](LICENSE)). Without a license key they run in evaluation mode;
licensed use needs a key in `MUNET_LICENSE`, `MUNET_LICENSE_FILE` or `~/.config/munet/license.key`. The Python and
Rust bindings are MIT licensed. To obtain a license, open an issue in this repository.

Every release lists `SHA256SUMS` for its files.
