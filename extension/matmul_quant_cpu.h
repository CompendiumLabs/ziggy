// matmul_quant includes

#include <torch/torch.h>

using namespace torch;

Tensor matmul_quant_cpu(Tensor a, Tensor b, unsigned int bits, float scale, float zero_point);
Tensor quantize_cpu(Tensor a, unsigned int bits, float scale, float zero_point);
Tensor dequantize_cpu(Tensor a, unsigned int bits, float scale, float zero_point, at::ScalarType typeb);
