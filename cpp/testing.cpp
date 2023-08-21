// matmul_quant testing

#include "matmul_quant.h"

void test_quantized() {
  at::IntArrayRef sizes = {4096, 256};
  Tensor a = torch::ones(sizes);
  Tensor b = torch::ones(sizes);

  Tensor qa = quantize_per_tensor(a, 2.0/128, 0, at::kQInt8);
  Tensor c = matmul_quant_float(qa.transpose(0, 1), b);

  std::cout << a.sizes() << " | " << a.mean() << std::endl;
  std::cout << b.sizes() << " | " << b.mean() << std::endl;
  std::cout << c.sizes() << " | " << c.mean() << std::endl;
}

int main() {
  test_quantized();
}
