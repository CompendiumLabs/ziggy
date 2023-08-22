// matmul_quant includes

#include <torch/torch.h>

using namespace torch;

Tensor matmul_qint8_float(Tensor a, Tensor b);
