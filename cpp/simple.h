#include <torch/torch.h>

using namespace torch;

Tensor pack(Tensor a, double scale, int64_t zero_point);
