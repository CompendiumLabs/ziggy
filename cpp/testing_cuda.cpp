// matmul_quant_cuda testing

#include "matmul_quant_cuda.h"

constexpr int64_t dim = 384;
constexpr int64_t n = 1048576;
constexpr int64_t m = 16;

void test_quantized_cuda() {
  int64_t dim = 384;
  int64_t n = 1048576;
  int64_t m = 16;

  Tensor a = torch::ones({n, dim}, at::device(torch::kCUDA).dtype(torch::kFloat));
  Tensor b = torch::ones({m, dim}, at::device(torch::kCUDA).dtype(torch::kHalf));

  Tensor qa = quantize_per_tensor(a, 4.0/128, 127, at::kQUInt8);
  Tensor c = matmul_quant_cuda(qa, b.transpose(0, 1));

  std::cout << a.sizes() << " | " << a.mean() << std::endl;
  std::cout << b.sizes() << " | " << b.mean() << std::endl;
  std::cout << c.sizes() << " | " << c.mean() << std::endl;
}

void test_pack_cuda() {
  int64_t dim = 8;
  int64_t n = 4;

  unsigned int bits = 4;
  float scale = 0.5f;
  float zero_point = 7.0f;

  Tensor a = torch::ones({n, dim}, at::device(torch::kCUDA).dtype(torch::kFloat));
  Tensor qa = quantize_and_pack(a, bits, scale, zero_point);

  std::cout << a.sizes() << " | " << a.mean() << std::endl;
  std::cout << qa.sizes() << std::endl << qa << std::endl;
}

int main() {
  test_quantized_cuda();
  std::cout << std::endl;
  test_pack_cuda();
}
