// matmul_quant extension

#include <torch/torch.h>

#include "matmul_quant_cpu.h"
#include "matmul_quant_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_and_pack_cpu", &quantize_and_pack_cpu, "Quantized and Pack CPU");
  m.def("quantize_and_pack_cuda", &quantize_and_pack_cuda, "Quantized and Pack CUDA");
  m.def("matmul_quant_cpu", &matmul_quant_cpu, "Quantized Matmul CPU");
  m.def("matmul_quant_cuda", &matmul_quant_cuda, "Quantized Matmul CUDA");
}
