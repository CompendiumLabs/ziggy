// matmul_quant_cuda testing

#include <string>
#include <vector>

#include "matmul_quant_cuda.h"

const int64_t dim = 8;
const int64_t n = 4;
const int64_t m = 8;

int main(int argc, char ** argv) {
    // default params
    int bits;
    float scale;
    float zero_point;

    // get number of bits and scale
    if (argc > 1) {
        bits = std::stoi(argv[1]);
    } else {
        bits = 8;
    }
    if (argc > 2) {
        scale = std::stof(argv[2]);
    } else {
        scale = 1.0f;
    }
    if (argc > 3) {
        zero_point = std::stof(argv[3]);
    } else {
        zero_point = float(1 << (bits - 1)) - 0.5;
    }

    std::cout << "Params" << std::endl;
    std::cout << "bits: " << bits << std::endl;
    std::cout << "scale: " << scale << std::endl;
    std::cout << "zero_point: " << zero_point << std::endl;
    std::cout << std::endl;

    // make first tensor
    std::vector<float> av(n * dim, 0.1f);
    Tensor a0 = torch::from_blob(av.data(), {n, dim}, at::device(torch::kCPU).dtype(torch::kFloat));
    Tensor a = a0.to(at::device(torch::kCUDA));
    std::cout << "Original:" << std::endl;
    std::cout << a << std::endl;
    std::cout << std::endl;

    // make second tensor
    Tensor b = torch::ones({m, dim}, at::device(torch::kCUDA).dtype(torch::kFloat));

    // quantize and pack
    Tensor qa = quantize_cuda(a, bits, scale, zero_point);
    std::cout << "Quantized:" << std::endl;
    std::cout << qa << std::endl;
    std::cout << std::endl;

    // dequantize and unpack
    Tensor a1 = dequantize_cuda(qa, torch::kFloat, bits, scale, zero_point);
    std::cout << "Dequantized:" << std::endl;
    std::cout << a1 << std::endl;
    std::cout << std::endl;

    // matmul results
    Tensor c = matmul_quant_cuda(qa, b.transpose(0, 1), bits, scale, zero_point);
    std::cout << "Matmul:" << std::endl;
    std::cout << c << std::endl;
}
