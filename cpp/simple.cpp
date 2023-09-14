#include "simple.h"

int main() {
    int64_t n = 8;
    Tensor a = torch::ones({n}, at::device(torch::kCUDA).dtype(torch::kFloat));
    Tensor b = pack(a, 0.5f, 0.0f);
    std::cout << b << std::endl;
}
