// matmul_quant includes

#include <torch/torch.h>

using namespace torch;

Tensor quantize_cuda(Tensor a, unsigned int bits, float scale, float zero_point);
Tensor dequantize_cuda(Tensor a, unsigned int bits, float scale, float zero_point, at::ScalarType typeb);
Tensor matmul_float_cuda(Tensor a, Tensor b);
Tensor matmul_quant_cuda(Tensor a, Tensor b, unsigned int bits, float scale, float zero_point);
