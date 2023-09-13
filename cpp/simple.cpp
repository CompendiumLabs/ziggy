#include "simple.h"

int main() {
    int64_t n = 8;
    Tensor a = torch::ones({n}, at::device(torch::kCUDA).dtype(torch::kFloat));
    Tensor b = pack(a, 0.5, 0);
    std::cout << b << std::endl;
}
