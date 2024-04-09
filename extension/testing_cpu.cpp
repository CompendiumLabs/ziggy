// matmul_quant_cpu testing

#include "matmul_quant_cpu.h"

void test_quantized_cpu() {
    int64_t dim = 8;
    int64_t n = 4;
    int64_t m = 4;

    Tensor a = torch::ones({n, dim}, at::device(torch::kCPU).dtype(torch::kFloat));
    Tensor b = torch::ones({m, dim}, at::device(torch::kCPU).dtype(torch::kFloat));

    unsigned int bits = 4;
    float scale = 1.0f;
    float zero_point = 7.0f;

    Tensor qa = quantize_and_pack_cpu(a, bits, scale, zero_point);
    Tensor c = matmul_quant_cpu(qa, b.transpose(0, 1), bits, scale, zero_point);

    std::cout << a.sizes() << std::endl << a << std::endl;
    std::cout << b.sizes() << std::endl << b << std::endl;
    std::cout << qa.sizes() << std::endl << qa << std::endl;
    std::cout << c.sizes() << std::endl << c << std::endl;
}

void test_pack_cpu() {
    int64_t dim = 8;
    int64_t n = 4;

    unsigned int bits = 4;
    float scale = 1.0f;
    float zero_point = 7.0f;

    Tensor a = torch::ones({n, dim}, at::device(torch::kCPU).dtype(torch::kFloat));
    Tensor qa = quantize_and_pack_cpu(a, bits, scale, zero_point);

    std::cout << a.sizes() << std::endl << a << std::endl;
    std::cout << qa.sizes() << std::endl << qa << std::endl;
}

void test_unpack_cpu() {
    int64_t dim = 8;
    int64_t n = 4;

    unsigned int bits = 4;
    float scale = 1.0f;
    float zero_point = 7.0f;

    Tensor a = torch::ones({n, dim}, at::device(torch::kCPU).dtype(torch::kFloat));
    Tensor qa = quantize_and_pack_cpu(a, bits, scale, zero_point);
    Tensor a1 = dequantize_and_unpack_cpu(qa, torch::kFloat, bits, scale, zero_point);

    std::cout << a.sizes() << std::endl << a << std::endl;
    std::cout << qa.sizes() << std::endl << qa << std::endl;
    std::cout << a1.sizes() << std::endl << a1 << std::endl;
}

int main() {
    test_quantized_cpu();
    std::cout << std::endl;
    test_pack_cpu();
    std::cout << std::endl;
    test_unpack_cpu();
}
