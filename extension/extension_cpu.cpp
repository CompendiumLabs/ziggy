// matmul_quant extension

#include <torch/torch.h>

#include "matmul_quant_cpu.h"

at::ScalarType dtype_string_to_scalar_type(const std::string& dtype_str) {
    if (dtype_str == "half") return at::kHalf;
    if (dtype_str == "float") return at::kFloat;
    throw std::runtime_error("Unsupported dtype");
}

Tensor quantize(Tensor a, unsigned int bits, float scale, float zero_point) {
    at::Device device = a.device();
    if (device.is_cuda()) {
        TORCH_CHECK(false, "ziggy extension does not support CUDA");
    } else if (device.is_cpu()) {
        return quantize_cpu(a, bits, scale, zero_point);
    } else {
        TORCH_CHECK(false, "quantize not implemented for '", device, "'");
    }
}

// there is already a torch.dequantize
Tensor dequantize1(Tensor a, unsigned int bits, float scale, float zero_point, const std::string& typeb_str) {
    at::ScalarType typeb = dtype_string_to_scalar_type(typeb_str);
    at::Device device = a.device();
    if (device.is_cuda()) {
        TORCH_CHECK(false, "ziggy extension does not support CUDA");
    } else if (device.is_cpu()) {
        return dequantize_cpu(a, bits, scale, zero_point, typeb);
    } else {
        TORCH_CHECK(false, "dequantize not implemented for '", device, "'");
    }
}

Tensor matmul_float(Tensor a, Tensor b) {
    at::Device device = a.device();
    if (device.is_cuda()) {
       TORCH_CHECK(false, "ziggy extension does not support CUDA");
    } else if (device.is_cpu()) {
        return matmul_float_cpu(a, b);
    } else {
        TORCH_CHECK(false, "matmul_float not implemented for '", device, "'");
    }
}

Tensor matmul_quant(Tensor a, Tensor b, unsigned int bits, float scale, float zero_point) {
    at::Device device = a.device();
    if (device.is_cuda()) {
        TORCH_CHECK(false, "ziggy extension does not support CUDA");
    } else if (device.is_cpu()) {
        return matmul_quant_cpu(a, b, bits, scale, zero_point);
    } else {
        TORCH_CHECK(false, "matmul_quant not implemented for '", device, "'");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize", &quantize, "Quantize");
    m.def("dequantize", &dequantize1, "Dequantize");
    m.def("matmul_float", &matmul_float, "Floating Matmul");
    m.def("matmul_quant", &matmul_quant, "Quantized Matmul");
}
