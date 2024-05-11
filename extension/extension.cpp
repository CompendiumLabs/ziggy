// matmul_quant extension

#include <torch/torch.h>

#include "matmul_quant_cpu.h"
#include "matmul_quant_cuda.h"

at::ScalarType dtype_string_to_scalar_type(const std::string& dtype_str) {
  if (dtype_str == "half") return at::kHalf;
  if (dtype_str == "float") return at::kFloat;
  throw std::runtime_error("Unsupported dtype");
}

Tensor quantize(Tensor a, unsigned int bits, float scale, float zero_point) {
  at::Device device = a.device();
  if (device.is_cuda()) {
    return quantize_cuda(a, bits, scale, zero_point);
  } else if (device.is_cpu()) {
    return quantize_cpu(a, bits, scale, zero_point);
  } else {
    TORCH_CHECK(false, "quantize_and_pack not implemented for '", device, "'");
  }
}

Tensor dequantize(Tensor a, const std::string& typeb_str, unsigned int bits, float scale, float zero_point) {
  at::ScalarType typeb = dtype_string_to_scalar_type(typeb_str);
  at::Device device = a.device();
  if (device.is_cuda()) {
    return dequantize_cuda(a, typeb, bits, scale, zero_point);
  } else if (device.is_cpu()) {
    return dequantize_cpu(a, typeb, bits, scale, zero_point);
  } else {
    TORCH_CHECK(false, "quantize_and_pack not implemented for '", device, "'");
  }
}

Tensor matmul_quant(Tensor a, Tensor b, unsigned int bits, float scale, float zero_point) {
  at::Device device = a.device();
  if (device.is_cuda()) {
    return matmul_quant_cuda(a, b, bits, scale, zero_point);
  } else if (device.is_cpu()) {
    return matmul_quant_cpu(a, b, bits, scale, zero_point);
  } else {
    TORCH_CHECK(false, "matmul_quant not implemented for '", device, "'");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", &quantize_and_pack, "Quantized and Pack");
  m.def("dequantize", &dequantize_and_unpack, "Dequantized and Unpack");
  m.def("matmul_quant", &matmul_quant, "Quantized Matmul");
}
