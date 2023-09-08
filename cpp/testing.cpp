// matmul_quant testing

#include "matmul_quant.h"

constexpr int64_t dim = 384;
constexpr int64_t n = 1048576;
constexpr int64_t m = 16;

void test_quantized_cpu() {
  Tensor a = torch::ones({n, dim});
  Tensor b = torch::ones({m, dim});

  Tensor qa = quantize_per_tensor(a, 4.0/128, 0, at::kQInt8);
  Tensor c = matmul_qint8_float_cpu(qa, b.transpose(0, 1));

  std::cout << a.sizes() << " | " << a.mean() << std::endl;
  std::cout << b.sizes() << " | " << b.mean() << std::endl;
  std::cout << c.sizes() << " | " << c.mean() << std::endl;
}

void test_quantized_cuda() {
  int64_t dim = 384;
  int64_t n = 1048576;
  int64_t m = 16;

  Tensor a = torch::ones({n, dim}, at::device(torch::kCUDA).dtype(torch::kFloat));
  Tensor b = torch::ones({m, dim}, at::device(torch::kCUDA).dtype(torch::kHalf));

  Tensor qa = quantize_per_tensor(a, 4.0/128, 0, at::kQInt8);
  Tensor c = matmul_qint8_cuda(qa, b.transpose(0, 1));

  std::cout << a.sizes() << " | " << a.mean() << std::endl;
  std::cout << b.sizes() << " | " << b.mean() << std::endl;
  std::cout << c.sizes() << " | " << c.mean() << std::endl;
}

int main() {
  test_quantized_cpu();
  std::cout << std::endl;
  test_quantized_cuda();
}
