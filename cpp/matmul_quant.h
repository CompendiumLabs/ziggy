// matmul_quant includes

#include <torch/torch.h>

using namespace torch;

Tensor matmul_qint8_float32_cpu(Tensor a, Tensor b);
Tensor matmul_qint8_float16_cuda(Tensor a, Tensor b);
