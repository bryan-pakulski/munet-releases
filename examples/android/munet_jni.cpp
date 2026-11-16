// JNI bridge between MuNet.kt and the MuNet C API (libmunet_c.so). See docs/android.md.
#include <android/log.h>
#include <jni.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "munet/munet_c.h"

namespace {

// UTF-8 view of a Java string ("" for null).
struct Chars {
  JNIEnv* env;
  jstring string;
  const char* chars;
  Chars(JNIEnv* e, jstring s) : env(e), string(s), chars(s ? e->GetStringUTFChars(s, nullptr) : nullptr) {}
  ~Chars() {
    if (chars) env->ReleaseStringUTFChars(string, chars);
  }
  const char* get() const { return chars ? chars : ""; }
};

// The C API reports failures through munet_last_error() (per thread); rethrow them as IllegalStateException.
void fail(JNIEnv* env, const char* what) {
  const std::string message = std::string(what) + ": " + munet_last_error();
  __android_log_print(ANDROID_LOG_ERROR, "MuNet", "%s", message.c_str());
  env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), message.c_str());
}

munet_program* program(jlong handle) { return reinterpret_cast<munet_program*>(handle); }

size_t elements(const std::vector<int64_t>& dims) {
  size_t n = 1;
  for (int64_t d : dims) n *= static_cast<size_t>(d);
  return n;
}

std::vector<int64_t> shape(const munet_program* p, size_t index, bool input) {
  const int32_t rank = input ? munet_input_shape(p, index, nullptr, 0) : munet_output_shape(p, index, nullptr, 0);
  std::vector<int64_t> dims(rank > 0 ? rank : 0);
  if (input) munet_input_shape(p, index, dims.data(), dims.size());
  else munet_output_shape(p, index, dims.data(), dims.size());
  return dims;
}

}  // namespace

extern "C" {

// Sets the kernel cache directory and, when given, the license key. Call once, before anything else.
// Returns the license status as JSON (the library's evaluation notice goes to stderr, which Android discards).
JNIEXPORT jstring JNICALL Java_com_example_munet_MuNet_nativeInit(JNIEnv* env, jclass, jstring cacheDir, jstring licenseKey) {
  Chars cache(env, cacheDir), key(env, licenseKey);
  if (*key.get()) setenv("MUNET_LICENSE", key.get(), 1);  // the library reads it once, on first use
  if (munet_set_cache_dir(cache.get()) != 0) {
    fail(env, "munet_set_cache_dir");
    return nullptr;
  }
  char json[1024];
  if (munet_license_json(json, sizeof json) < 0) {
    fail(env, "munet_license_json");
    return nullptr;
  }
  return env->NewStringUTF(json);
}

// Checks and times the per-GPU kernel choices and stores a device profile in the cache directory. Takes seconds:
// call it off the UI thread, once per install (and again after a system update). Returns the profile as JSON.
JNIEXPORT jstring JNICALL Java_com_example_munet_MuNet_nativeProbe(JNIEnv* env, jclass, jint device) {
  if (munet_probe_device(device) != 0) {
    fail(env, "munet_probe_device");
    return nullptr;
  }
  const int32_t length = munet_device_profile_json(device, nullptr, 0);
  if (length < 0) {
    fail(env, "munet_device_profile_json");
    return nullptr;
  }
  std::string json(static_cast<size_t>(length) + 1, '\0');
  munet_device_profile_json(device, json.data(), json.size());
  return env->NewStringUTF(json.c_str());
}

// Loads model.onnx or model.mnet from a file path (copy APK assets to filesDir first). device -1: CPU executor.
JNIEXPORT jlong JNICALL Java_com_example_munet_MuNet_nativeLoad(JNIEnv* env, jclass, jstring path, jint device, jboolean fp16) {
  Chars file(env, path);
  munet_options options = munet_default_options();
  options.device = device;
  options.precision = fp16 ? MUNET_PRECISION_FP16 : MUNET_PRECISION_FP32;
  const std::string name = file.get();
  const bool onnx = name.size() > 5 && name.compare(name.size() - 5, 5, ".onnx") == 0;
  munet_program* p = onnx ? munet_load_onnx(file.get(), nullptr, &options) : munet_load(file.get(), &options);
  if (!p) fail(env, "load");
  return reinterpret_cast<jlong>(p);
}

JNIEXPORT void JNICALL Java_com_example_munet_MuNet_nativeFree(JNIEnv*, jclass, jlong handle) { munet_free(program(handle)); }

// Runs the program on one float array per input, in declaration order.
JNIEXPORT void JNICALL Java_com_example_munet_MuNet_nativeRun(JNIEnv* env, jclass, jlong handle, jobjectArray inputs) {
  munet_program* p = program(handle);
  const size_t count = munet_input_count(p);
  if (static_cast<size_t>(env->GetArrayLength(inputs)) != count) {
    env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "wrong number of inputs");
    return;
  }
  std::vector<std::vector<float>> data(count);
  std::vector<const float*> pointers(count);
  for (size_t i = 0; i < count; ++i) {
    auto array = static_cast<jfloatArray>(env->GetObjectArrayElement(inputs, static_cast<jsize>(i)));
    data[i].resize(elements(shape(p, i, true)));
    if (!array || static_cast<size_t>(env->GetArrayLength(array)) != data[i].size()) {
      env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "input size does not match the model");
      return;
    }
    env->GetFloatArrayRegion(array, 0, static_cast<jsize>(data[i].size()), data[i].data());
    env->DeleteLocalRef(array);
    pointers[i] = data[i].data();
  }
  if (munet_run(p, pointers.data()) != 0) fail(env, "munet_run");
}

JNIEXPORT jlongArray JNICALL Java_com_example_munet_MuNet_nativeOutputShape(JNIEnv* env, jclass, jlong handle, jint index) {
  const auto dims = shape(program(handle), static_cast<size_t>(index), false);
  jlongArray result = env->NewLongArray(static_cast<jsize>(dims.size()));
  std::vector<jlong> values(dims.begin(), dims.end());
  env->SetLongArrayRegion(result, 0, static_cast<jsize>(values.size()), values.data());
  return result;
}

// Output `index` of the last run.
JNIEXPORT jfloatArray JNICALL Java_com_example_munet_MuNet_nativeOutput(JNIEnv* env, jclass, jlong handle, jint index) {
  munet_program* p = program(handle);
  std::vector<float> data(elements(shape(p, static_cast<size_t>(index), false)));
  if (munet_read_output(p, static_cast<size_t>(index), data.data(), data.size()) != 0) {
    fail(env, "munet_read_output");
    return nullptr;
  }
  jfloatArray result = env->NewFloatArray(static_cast<jsize>(data.size()));
  env->SetFloatArrayRegion(result, 0, static_cast<jsize>(data.size()), data.data());
  return result;
}

}  // extern "C"
