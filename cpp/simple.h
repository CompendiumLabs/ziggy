#include <torch/torch.h>

using namespace torch;

Tensor pack(Tensor a, float scale, float zero_point);
