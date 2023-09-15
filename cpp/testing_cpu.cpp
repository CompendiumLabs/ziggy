// matmul_quant_cpu testing

#include "matmul_quant_cpu.h"

constexpr int64_t dim = 384;
constexpr int64_t n = 1048576;
constexpr int64_t m = 16;

void test_quantized_cpu() {
  Tensor a = torch::ones({n, dim});
  Tensor b = torch::ones({m, dim});

  unsigned int bits = 8;
  float scale = 1.0f;
  float zero_point = 127.0f;

  Tensor qa = quantize_and_pack_cpu(a, bits, scale, zero_point);
  Tensor c = matmul_quant_cpu(qa, b.transpose(0, 1), bits, scale, zero_point);

  std::cout << a.sizes() << " | " << a.mean() << std::endl;
  std::cout << b.sizes() << " | " << b.mean() << std::endl;
  std::cout << c.sizes() << " | " << c.mean() << std::endl;
}

int main() {
  test_quantized_cpu();
  std::cout << std::endl;
}
