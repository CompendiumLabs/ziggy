// matmul_quant extension

#include <torch/torch.h>

#include "matmul_quant_cpu.h"
#include "matmul_quant_cuda.h"

Tensor quantize_and_pack(Tensor a, unsigned int bits, float scale, float zero_point) {
  at::Device device = a.device();
  if (device.is_cuda()) {
    return quantize_and_pack_cuda(a, bits, scale, zero_point);
  } else if (device.is_cpu()) {
    return quantize_and_pack_cpu(a, bits, scale, zero_point);
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
  m.def("quantize_and_pack", &quantize_and_pack, "Quantized and Pack");
  m.def("matmul_quant", &matmul_quant, "Quantized Matmul");
}
