// matmul_quant extension

#include <torch/torch.h>

#include "matmul_quant.h"

using namespace torch;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_quant_float", &matmul_quant_float, "Quantized Matmul");
}
