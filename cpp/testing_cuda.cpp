// matmul_quant_cuda testing

#include "matmul_quant_cuda.h"

constexpr int dim = 384;
constexpr int n = 1048576;
constexpr int m = 16;

void test_quantized_cuda() {
  int dim = 8;
  int n = 4;
  int m = 4;

  Tensor a = torch::ones({n, dim}, at::device(torch::kCUDA).dtype(torch::kFloat));
  Tensor b = torch::ones({m, dim}, at::device(torch::kCUDA).dtype(torch::kFloat));

  unsigned int bits = 4;
  float scale = 1.0f;
  float zero_point = 7.0f;

  Tensor qa = quantize_and_pack_cuda(a, bits, scale, zero_point);
  Tensor c = matmul_quant_cuda(qa, b.transpose(0, 1), bits, scale, zero_point);

  std::cout << a.sizes() << std::endl << a << std::endl;
  std::cout << b.sizes() << std::endl << b << std::endl;
  std::cout << qa.sizes() << std::endl << qa << std::endl;
  std::cout << c.sizes() << std::endl << c << std::endl;
}

void test_pack_cuda() {
  int dim = 8;
  int n = 4;

  unsigned int bits = 4;
  float scale = 1.0f;
  float zero_point = 7.0f;

  Tensor a = torch::ones({n, dim}, at::device(torch::kCUDA).dtype(torch::kFloat));
  Tensor qa = quantize_and_pack_cuda(a, bits, scale, zero_point);

  std::cout << a.sizes() << " | " << a.mean() << std::endl;
  std::cout << qa.sizes() << std::endl << qa << std::endl;
}

int main() {
  test_quantized_cuda();
  std::cout << std::endl;
  test_pack_cuda();
}
