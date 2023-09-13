// matmul_quant includes

#include <torch/torch.h>

using namespace torch;

Tensor matmul_qint8_float_cpu(Tensor a, Tensor b);
Tensor matmul_qint8_cuda(Tensor a, Tensor b);
Tensor quantize_and_pack(Tensor a, unsigned int bits, double scale, int64_t zero_point);
