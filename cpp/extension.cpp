// matmul_quant extension

#include <torch/torch.h>

#include "matmul_quant.h"

using namespace torch;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_qint8_float32_cpu", &matmul_qint8_float32_cpu, "Quantized Matmul CPU");
  m.def("matmul_qint8_float16_cuda", &matmul_qint8_float16_cuda, "Quantized Matmul CUDA");
}
