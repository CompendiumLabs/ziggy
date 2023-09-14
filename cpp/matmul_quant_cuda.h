// matmul_quant includes

#include <torch/torch.h>

using namespace torch;

Tensor matmul_quant_cuda(Tensor a, Tensor b, unsigned int bits, float scale, float zero_point);
Tensor quantize_and_pack(Tensor a, unsigned int bits, float scale, float zero_point);
