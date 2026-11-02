// Minimal C++ client of the MuNet C API: load a .mnet program, describe it, run it on zeros and print each output's
// first values and the GPU time. See docs/c.md.
//
//   c++ -std=c++17 run_model.cpp -I<sdk>/include -L<sdk>/lib -lmunet_c -o run_model   # shared library
//   ./run_model model.mnet [device]          # device -1: CPU reference executor
#include "munet/munet_c.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// RAII ownership and error translation; the C API reports failures through munet_last_error() (per thread).
struct ProgramDeleter {
  void operator()(munet_program* p) const { munet_free(p); }
};
using Program = std::unique_ptr<munet_program, ProgramDeleter>;

void check(int32_t rc, const char* what) {
  if (rc != 0) throw std::runtime_error(std::string(what) + ": " + munet_last_error());
}

std::vector<int64_t> shape(const munet_program* p, size_t index, bool input) {
  int32_t rank = input ? munet_input_shape(p, index, nullptr, 0) : munet_output_shape(p, index, nullptr, 0);
  std::vector<int64_t> dims(rank > 0 ? rank : 0);
  if (input) munet_input_shape(p, index, dims.data(), dims.size());
  else munet_output_shape(p, index, dims.data(), dims.size());
  return dims;
}

size_t elements(const std::vector<int64_t>& dims) {
  size_t n = 1;
  for (int64_t d : dims) n *= static_cast<size_t>(d);
  return n;
}

std::string describe(const std::vector<int64_t>& dims) {
  std::string s = "[";
  for (size_t i = 0; i < dims.size(); ++i) s += (i ? "," : "") + std::to_string(dims[i]);
  return s + "]";
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s model.mnet [device]\n", argv[0]);
    return 2;
  }
  std::printf("MuNet %s (C API %u), %d Vulkan device(s)\n", munet_version(), munet_api_version(), munet_device_count());

  munet_options options = munet_default_options();
  if (argc > 2) options.device = std::atoi(argv[2]);
  options.precision = MUNET_PRECISION_MIXED;
  Program program(munet_load(argv[1], &options));
  if (!program) throw std::runtime_error(std::string("load: ") + munet_last_error());

  // Inputs: contiguous float32 tensors in declaration order (integer inputs carry exact integer values as floats).
  std::vector<std::vector<float>> inputs;
  std::vector<const float*> pointers;
  for (size_t i = 0; i < munet_input_count(program.get()); ++i) {
    auto dims = shape(program.get(), i, true);
    std::printf("input  %s %s %s\n", munet_input_name(program.get(), i), munet_input_dtype(program.get(), i),
                describe(dims).c_str());
    inputs.emplace_back(elements(dims), 0.0f);
    pointers.push_back(inputs.back().data());
  }
  check(munet_run(program.get(), pointers.data()), "run");

  for (size_t i = 0; i < munet_output_count(program.get()); ++i) {
    auto dims = shape(program.get(), i, false);
    std::vector<float> data(elements(dims));
    check(munet_read_output(program.get(), i, data.data(), data.size()), "read");
    std::printf("output %s %s %s first=%g%s\n", munet_output_name(program.get(), i), munet_output_dtype(program.get(), i),
                describe(dims).c_str(), data.empty() ? 0.0 : data[0],
                munet_output_padded(program.get(), i) ? " (NMS rows, -1 padded)" : "");
  }
  double ms = munet_gpu_ms(program.get());
  if (ms >= 0) std::printf("GPU time %.3f ms\n", ms);
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
