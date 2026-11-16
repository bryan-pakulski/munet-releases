# MuNet on Android

MuNet runs on Android through its C API: one shared library, `libmunet_c.so`, that loads ONNX or `.mnet` models and
runs them on the phone's GPU through Vulkan. This page adds it to an Android Studio project.

## Requirements

- Android 10 (API 29) or later, 64-bit: `arm64-v8a` for phones, `x86_64` for the emulator.
- A Vulkan 1.1 GPU driver. The CPU reference executor (`device = -1`) needs none, but it is slow.
- The NDK and CMake 3.22 or later, installed from the Android Studio SDK manager.

## What the archive holds

Each release publishes `munet-android-<version>.tar.gz` on
[the release downloads](https://github.com/bryan-pakulski/munet-releases/releases):

```
include/munet/munet_c.h         # the C API
jniLibs/arm64-v8a/libmunet_c.so
jniLibs/x86_64/libmunet_c.so
examples/munet_jni.cpp          # JNI bridge
examples/CMakeLists.txt         # builds the bridge
examples/MuNet.kt               # Kotlin wrapper
docs/                           # the guides this page links to
third_party/                    # licenses of the libraries linked into libmunet_c.so
LICENSE, README.md
```

`libmunet_c.so` is about 9 MB per ABI. It exports only the `munet_*` functions, depends only on `libc`, `libm` and
`libdl`, and loads the system `libvulkan.so` on first GPU use. It compiles its kernels on the device, so the app needs
no shader compiler. Its segments are aligned for 16 KB pages.

## Add it to a project

1. Extract the archive into the app module as `app/munet`.
2. Copy `app/munet/examples/MuNet.kt` to `app/src/main/java/com/example/munet/MuNet.kt`.
3. Point Gradle at the bridge in `app/build.gradle.kts`:

```kotlin
android {
    defaultConfig {
        minSdk = 29
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
        externalNativeBuild {
            cmake { arguments += listOf("-DMUNET_ANDROID=${projectDir}/munet", "-DANDROID_STL=c++_static") }
        }
    }
    externalNativeBuild {
        cmake { path = file("munet/examples/CMakeLists.txt") }
    }
}
```

Gradle builds `libmunet_jni.so` and packages it with `libmunet_c.so`. Do not also list `munet/jniLibs` as a `jniLibs`
source directory: the library would be packaged twice and the build fails.

The bridge's function names contain the Kotlin package (`Java_com_example_munet_MuNet_...`). To use another package,
change the `package` line in `MuNet.kt` and rename the functions in `munet_jni.cpp` to match.

## Use it

```kotlin
// Once, at start-up. Returns the license status as JSON.
val license = MuNet.init(context, licenseKey = BuildConfig.MUNET_LICENSE)

// Once per install, off the UI thread: checks and times the kernel choices for this GPU.
MuNet.probe()

// Models are loaded from a file path, so copy an APK asset to filesDir first.
val file = File(context.filesDir, "model.onnx")
if (!file.exists()) context.assets.open("model.onnx").use { input -> file.outputStream().use(input::copyTo) }

MuNet.load(file.path).use { model ->          // off the UI thread
    model.run(pixels)                         // one FloatArray per model input, in declaration order
    val scores = model.output(0)              // FloatArray
    val shape = model.outputShape(0)          // LongArray
}
```

- **Threads.** Loading compiles kernels and can take seconds the first time. Load, probe and run on a background
  thread. Use one model object from one thread at a time.
- **Cache.** `MuNet.init` sets the kernel cache to `context.cacheDir`. Later loads reuse it. Without it, kernels are
  recompiled on every load.
- **Inputs.** Every input is a `FloatArray` in the model's shape, flattened. Integer inputs take whole numbers as floats.
- **Errors.** Failures throw `IllegalStateException` with the library's message, which is also written to logcat under
  the tag `MuNet`.

## Precision

`MuNet.load(path, fp16 = true)` is the default and the faster choice on phone GPUs. `fp16 = false` runs fp32, which
matches the CPU result to float rounding and is slower. See [inference.md](inference.md) for how precision works and
[quantization.md](quantization.md) for int8.

## The device probe

Phone GPU drivers differ in which kernel forms are correct and which are fast. `MuNet.probe()` checks MuNet's choices
against its CPU executor on this GPU, times the correct ones and stores the result in the cache directory. Later loads
use it. Run it once per install and again after a system update, since the profile is stored per driver version.
See [the device probe](inference.md#device-probe) and [mobile GPUs](gpus.md#mobile-and-embedded-arm-mali-qualcomm-adreno).

## License key

Without a key the library runs in evaluation mode. Its evaluation notice goes to stderr, which Android discards, so
check the JSON that `MuNet.init` returns: `status` is `licensed`, `evaluation`, `invalid` or `expired`.

Pass the key to `MuNet.init` before anything else. The library reads it once, on first use. With an invalid or expired
key, GPU execution fails. See [key setup](getting-started.md#license-key).

## Use the C API directly

`munet_jni.cpp` is a small example, not a required layer. Native code can include `munet/munet_c.h` and link
`libmunet_c.so` itself, for example to pass camera frames without copying them through Java arrays. The functions are
described in [the C guide](c.md), and [zero-copy.md](zero-copy.md) covers sharing a Vulkan device.

## Build the library yourself

In an engine checkout, `python tools/build_android.py --package dist` builds shaderc and `libmunet_c.so` for both ABIs
with the NDK and writes the archive. `--test` runs the parity check on a phone connected over `adb`: every test model
must give the same outputs on the CPU executor and the GPU.

## Troubleshooting

- **`UnsatisfiedLinkError: dlopen failed: library "libmunet_c.so" not found`:** the ABI is missing from `abiFilters`,
  or `-DMUNET_ANDROID` does not point at the extracted archive.
- **`No implementation found for ... nativeInit`:** the Kotlin package and the function names in `munet_jni.cpp`
  differ.
- **A load fails with a Vulkan error:** the device has no usable Vulkan 1.1 driver. `device = -1` runs on the CPU.
- **Wrong or non-finite outputs on a GPU that has not been probed:** run `MuNet.probe()` and load again. If the outputs
  are still wrong, see [troubleshooting.md](troubleshooting.md).
