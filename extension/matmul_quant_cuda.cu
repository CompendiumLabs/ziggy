#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <torch/torch.h>

/*
#include "macros.h"
DISPATCH_BITWIDTH(bits, [&] {
    quantize_and_pack_float<bit_width><<<blocks, threads>>>(
        a_ptr, b_ptr, sn, sk, tan, tak, scale, zero_point
    );
});
*/

using namespace torch;

// global constants

constexpr int64_t kWarpSize = 32;
constexpr int64_t MAX_GRID = 65535;

// half-float conversion utils

template <typename scalar1_t, typename scalar2_t>
__inline__ __device__ float multiply_float(scalar1_t x, scalar2_t y);

template <>
__inline__ __device__ float multiply_float(float x, float y) {
    return x * y;
}

template <>
__inline__ __device__ float multiply_float(float x, __half y) {
    return x * __half2float(y);
}

template <>
__inline__ __device__ float multiply_float(__half x, float y) {
    return __half2float(x) * y;
}

template <>
__inline__ __device__ float multiply_float(__half x, __half y) {
    return __half2float(x) * __half2float(y);
}

// quantization kernels

// we need to use unsigned intN packing here
template <unsigned int bits>
__global__ void quantize_float_kernel(float* a, uint8_t* b, int64_t sn, int64_t sk, int64_t tan, int64_t tak, float scale, float zero_point) {
    constexpr unsigned int qFact = 8 / bits;
    constexpr unsigned int pMax = (1 << bits) - 1;
    constexpr float pMax_f = (float)pMax;

    const int64_t sk_p = sk / qFact;

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < sn) {
        float* posa = a + i * tan;

        for (int64_t k = 0; k < sk_p; k++) {
            // init zero bits
            uint8_t valb = 0;

            // pack into packed_t-sized values
            for (int s = 0; s < 8; s += bits) {
                float vala = (*posa);
                vala = vala / scale + zero_point;
                vala = max(0.0f, min(pMax_f, vala));
                uint8_t vala_i = __float2uint_rn(vala);
                valb |= vala_i << s;
                posa += tak;
            }

            // store packed value
            b[i * sk_p + k] = valb;
        }
    }
}

template <unsigned int bits>
__global__ void quantize_half_kernel(__half* a, uint8_t* b, int64_t sn, int64_t sk, int64_t tan, int64_t tak, float scale, float zero_point) {
    constexpr unsigned int qFact = 8 / bits;
    constexpr unsigned int pMax = (1 << bits) - 1;

    const int64_t sk_p = sk / qFact;
    const __half scale_h = __float2half(scale);
    const __half zero_point_h = __float2half(zero_point);
    const __half zero_h = __float2half(0.0f);
    const __half pMax_h = __int2half_rn(pMax);

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < sn) {
        __half* posa = a + i * tan;

        for (int64_t k = 0; k < sk_p; k++) {
            // init zero bits
            uint8_t valb = 0;

            // pack into packed_t-sized values
            for (int s = 0; s < 8; s += bits) {
                __half vala = (*posa);
                vala = vala / scale_h + zero_point_h;
                vala = __hmax(zero_h, __hmin(pMax_h, vala));
                uint8_t vala_i = __half2int_rn(vala);
                valb |= vala_i << s;
                posa += tak;
            }

            // store packed value
            b[i * sk_p + k] = valb;
        }
    }
}

template <unsigned int bits>
__global__ void dequantize_float_kernel(uint8_t* a, float* b, int64_t sn, int64_t sk, float scale, float zero_point) {
    constexpr unsigned int qFact = 8 / bits;
    constexpr unsigned int pMax = (1 << bits) - 1;
    constexpr float pMax_f = (float)pMax;

    const int64_t sk_p = sk / qFact;

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < sn) {
        uint8_t* posa = a + i * sk_p;
        float* posb = b + i * sk;

        for (int64_t k = 0; k < sk_p; k++) {
            // load packed values
            uint8_t vala = (*posa);

            // unpack into bits-sized values
            for (int s = 0; s < 8; s += bits) {
                uint8_t vala_i = (uint8_t)((vala >> s) & pMax);
                float vala_f = (float)vala_i;
                (*posb) = scale * (vala_f - zero_point);
                posb++;
            }

            // increment base value
            posa++;
        }
    }
}

template <unsigned int bits>
__global__ void dequantize_half_kernel(uint8_t* a, __half* b, int64_t sn, int64_t sk, float scale, float zero_point) {
    constexpr unsigned int qFact = 8 / bits;
    constexpr unsigned int pMax = (1 << bits) - 1;

    const int64_t sk_p = sk / qFact;
    const __half scale_h = __float2half(scale);
    const __half zero_point_h = __float2half(zero_point);

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < sn) {
        uint8_t* posa = a + i * sk_p;
        __half* posb = b + i * sk;

        for (int64_t k = 0; k < sk_p; k++) {
            // load packed values
            uint8_t vala = (*posa);

            // unpack into bits-sized values
            for (int s = 0; s < 8; s += bits) {
                uint8_t vala_i = (uint8_t)((vala >> s) & pMax);
                __half vala_h = __int2half_rn(vala_i);
                (*posb) = scale_h * (vala_h - zero_point_h);
                posb++;
            }

            // increment base value
            posa++;
        }
    }
}

template <typename scalar1_t, typename scalar2_t>
__global__ void matmul_float_kernel(scalar1_t* a, scalar2_t* b, float* c, int64_t sn, int64_t sm, int64_t sk, int64_t tan, int64_t tak, int64_t tbk, int64_t tbm) {
    int64_t i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t j0 = blockIdx.y * blockDim.y + threadIdx.y;

    int64_t grid_stride_x = gridDim.x * blockDim.x;
    int64_t grid_stride_y = gridDim.y * blockDim.y;

    for (int64_t i = i0; i < sn; i += grid_stride_x) {
        for (int64_t j = j0; j < sm; j += grid_stride_y) {
            scalar1_t* posa = a + i * tan;
            scalar2_t* posb = b + j * tbm;
            float sum = 0.0f;

            for (int64_t k = 0; k < sk; k++) {
                scalar1_t vala = (*posa);
                scalar2_t valb = (*posb);
                sum += multiply_float<scalar1_t, scalar2_t>(vala, valb);
                posa += tak;
                posb += tbk;
            }

            c[i * sm + j] = sum;
        }
    }
}

template <unsigned int bits, typename scalar_t>
__global__ void matmul_quant_kernel(uint8_t* a, scalar_t* b, float* c, int64_t sn, int64_t sm, int64_t sk, int64_t tan, int64_t tak, int64_t tbk, int64_t tbm, float scale, float zero_point) {
    constexpr uint8_t mask = (1 << bits) - 1;

    int64_t i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t j0 = blockIdx.y * blockDim.y + threadIdx.y;

    int64_t grid_stride_x = gridDim.x * blockDim.x;
    int64_t grid_stride_y = gridDim.y * blockDim.y;

    for (int64_t i = i0; i < sn; i += grid_stride_x) {
        for (int64_t j = j0; j < sm; j += grid_stride_y) {
            uint8_t* posa = a + i * tan;
            scalar_t* posb = b + j * tbm;
            float sum = 0.0f;

            for (int64_t k = 0; k < sk; k++) {
                // load packed values
                uint8_t vala = (*posa);

                // unpack into bits-sized values
                for (int s = 0; s < 8; s += bits) {
                    uint8_t vala_i = (uint8_t)((vala >> s) & mask);
                    float vala_f = (float)vala_i - zero_point;
                    scalar_t valb_f = (*posb);
                    sum += multiply_float<float, scalar_t>(vala_f, valb_f);
                    posb += tbk;
                }

                // increment base value
                posa += tak;
            }

            c[i * sm + j] = scale * sum;
        }
    }
}

Tensor quantize_cuda(Tensor a, unsigned int bits, float scale, float zero_point) {
    at::ScalarType typea = a.scalar_type();
    at::IntArrayRef sizesa = a.sizes();
    at::IntArrayRef stridesa = a.strides();

    int64_t sn = sizesa[0];
    int64_t sk = sizesa[1];
    int64_t tan = stridesa[0];
    int64_t tak = stridesa[1];

    int64_t sk_p = sk / (8 / bits);
    int64_t tbm = sk_p;
    int64_t tbk = 1;

    dim3 threads(kWarpSize);
    dim3 blocks((sn + threads.x - 1) / threads.x);

    Tensor b = torch::empty({sn, sk_p}, torch::device(kCUDA).dtype(torch::kUInt8));
    uint8_t* b_ptr = b.data_ptr<uint8_t>();

    switch (typea) {
        case torch::kFloat: {
            float* a_ptr = a.data_ptr<float>();

            switch (bits) {
                case 8: {
                    quantize_float_kernel<8><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, tan, tak, scale, zero_point);
                    break;
                }
                case 4: {
                    quantize_float_kernel<4><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, tan, tak, scale, zero_point);
                    break;
                }
                case 2: {
                    quantize_float_kernel<2><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, tan, tak, scale, zero_point);
                    break;
                }
                case 1: {
                    quantize_float_kernel<1><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, tan, tak, scale, zero_point);
                    break;
                }
                default: {
                    TORCH_CHECK(false, "Unsupported number of quantization bits '", bits, "'");
                }
            }

            break;
        }
        case torch::kHalf: {
            __half* a_ptr = reinterpret_cast<__half*>(a.data_ptr<at::Half>());

            switch (bits) {
                case 8: {
                    quantize_half_kernel<8><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, tan, tak, scale, zero_point);
                    break;
                }
                case 4: {
                    quantize_half_kernel<4><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, tan, tak, scale, zero_point);
                    break;
                }
                case 2: {
                    quantize_half_kernel<2><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, tan, tak, scale, zero_point);
                    break;
                }
                case 1: {
                    quantize_half_kernel<1><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, tan, tak, scale, zero_point);
                    break;
                }
                default: {
                    TORCH_CHECK(false, "Unsupported number of quantization bits '", bits, "'");
                }
            }

            break;
        }
        default: {
            TORCH_CHECK(false, "Unsupported type for input tensor '", typea, "'");
        }
    }

    return b;
}

// this assumes normal strides since `a` is packed anyway
Tensor dequantize_cuda(Tensor a, unsigned int bits, float scale, float zero_point, at::ScalarType typeb) {
    at::ScalarType typea = a.scalar_type();
    at::IntArrayRef sizesa = a.sizes();
    assert(typea == torch::kUInt8);

    int64_t sn = sizesa[0];
    int64_t sk_p = sizesa[1];
    int64_t sk = sk_p * (8 / bits);

    dim3 threads(kWarpSize);
    dim3 blocks((sn + threads.x - 1) / threads.x);

    Tensor b = torch::empty({sn, sk}, torch::device(kCUDA).dtype(typeb));
    uint8_t* a_ptr = a.data_ptr<uint8_t>();

    switch (typeb) {
        case torch::kFloat: {
            float* b_ptr = b.data_ptr<float>();

            switch (bits) {
                case 8: {
                    dequantize_float_kernel<8><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, scale, zero_point);
                    break;
                }
                case 4: {
                    dequantize_float_kernel<4><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, scale, zero_point);
                    break;
                }
                case 2: {
                    dequantize_float_kernel<2><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, scale, zero_point);
                    break;
                }
                case 1: {
                    dequantize_float_kernel<1><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, scale, zero_point);
                    break;
                }
                default: {
                    TORCH_CHECK(false, "Unsupported number of quantization bits '", bits, "'");
                }
            }

            break;
        }
        case torch::kHalf: {
            __half* b_ptr = reinterpret_cast<__half*>(b.data_ptr<at::Half>());

            switch (bits) {
                case 8: {
                    dequantize_half_kernel<8><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, scale, zero_point);
                    break;
                }
                case 4: {
                    dequantize_half_kernel<4><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, scale, zero_point);
                    break;
                }
                case 2: {
                    dequantize_half_kernel<2><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, scale, zero_point);
                    break;
                }
                case 1: {
                    dequantize_half_kernel<1><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, scale, zero_point);
                    break;
                }
                default: {
                    TORCH_CHECK(false, "Unsupported number of quantization bits '", bits, "'");
                }
            }

            break;
        }
        default: {
            TORCH_CHECK(false, "Unsupported type for input tensor '", typeb, "'");
        }
    }

    return b;
}

// always uses float32 accumulation
Tensor matmul_float_cuda(Tensor a, Tensor b) {
    at::Device devicea = a.device();
    at::Device deviceb = b.device();

    at::ScalarType typea = a.scalar_type();
    at::ScalarType typeb = b.scalar_type();

    at::IntArrayRef sizesa = a.sizes();
    at::IntArrayRef sizesb = b.sizes();
    at::IntArrayRef stridesa = a.strides();
    at::IntArrayRef stridesb = b.strides();

    assert(devicea == deviceb);
    assert(sizesa[1] == sizesb[0]);

    int64_t sn = sizesa[0];
    int64_t sm = sizesb[1];
    int64_t sk = sizesa[1];
    int64_t tan = stridesa[0];
    int64_t tak = stridesa[1];
    int64_t tbk = stridesb[0];
    int64_t tbm = stridesb[1];

    dim3 threads(kWarpSize, kWarpSize);
    dim3 blocks(
        min(MAX_GRID, (sn + threads.x - 1) / threads.x),
        min(MAX_GRID, (sm + threads.y - 1) / threads.y)
    );

    Tensor c = at::empty({sn, sm}, at::device(kCUDA).dtype(torch::kFloat));
    float* c_ptr = c.data_ptr<float>();

    switch (typea) {
        case torch::kHalf: {
            __half* a_ptr = reinterpret_cast<__half*>(a.data_ptr<at::Half>());

            switch (typeb) {
                case torch::kHalf: {
                    __half* b_ptr = reinterpret_cast<__half*>(b.data_ptr<at::Half>());

                    matmul_float_kernel<__half, __half><<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, sn, sm, sk, tan, tak, tbk, tbm);
                    break;
                }
                case torch::kFloat: {
                    float* b_ptr = b.data_ptr<float>();

                    matmul_float_kernel<__half, float><<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, sn, sm, sk, tan, tak, tbk, tbm);
                    break;
                }
                default: {
                    TORCH_CHECK(false, "Unsupported type for comparison tensor '", typeb, "'");
                }
            }
            break;
        }
        case torch::kFloat: {
            float* a_ptr = a.data_ptr<float>();

            switch (typeb) {
                case torch::kHalf: {
                    __half* b_ptr = reinterpret_cast<__half*>(b.data_ptr<at::Half>());

                    matmul_float_kernel<float, __half><<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, sn, sm, sk, tan, tak, tbk, tbm);
                    break;
                }
                case torch::kFloat: {
                    float* b_ptr = b.data_ptr<float>();

                    matmul_float_kernel<float, float><<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, sn, sm, sk, tan, tak, tbk, tbm);
                    break;
                }
                default: {
                    TORCH_CHECK(false, "Unsupported type for comparison tensor '", typeb, "'");
                }
            }
            break;
        }
        default: {
            TORCH_CHECK(false, "Unsupported type for input tensor '", typea, "'");
        }
    }

    return c;
}

Tensor matmul_quant_cuda(Tensor a, Tensor b, unsigned int bits, float scale, float zero_point) {
    at::Device devicea = a.device();
    at::Device deviceb = b.device();

    at::ScalarType typea = a.scalar_type();
    at::ScalarType typeb = b.scalar_type();

    at::IntArrayRef sizesa = a.sizes();
    at::IntArrayRef sizesb = b.sizes();
    at::IntArrayRef stridesa = a.strides();
    at::IntArrayRef stridesb = b.strides();

    assert(devicea == deviceb);
    assert(typea == torch::kUInt8);
    assert((8 / bits) * sizesa[1] == sizesb[0]);

    int64_t sn = sizesa[0];
    int64_t sm = sizesb[1];
    int64_t sk = sizesa[1];
    int64_t tan = stridesa[0];
    int64_t tak = stridesa[1];
    int64_t tbk = stridesb[0];
    int64_t tbm = stridesb[1];

    Tensor c = at::empty({sn, sm}, at::device(kCUDA).dtype(torch::kFloat));

    uint8_t* a_ptr = a.data_ptr<uint8_t>();
    float* c_ptr = c.data_ptr<float>();

    dim3 threads(kWarpSize, kWarpSize);
    dim3 blocks(
        min(MAX_GRID, (sn + threads.x - 1) / threads.x),
        min(MAX_GRID, (sm + threads.y - 1) / threads.y)
    );

    switch (typeb) {
        case at::kHalf: {
            __half* b_ptr = reinterpret_cast<__half*>(b.data_ptr<at::Half>());

            switch (bits) {
                case 8: {
                    matmul_quant_kernel<8, half><<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, sn, sm, sk, tan, tak, tbk, tbm, scale, zero_point);
                    break;
                }
                case 4: {
                    matmul_quant_kernel<4, half><<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, sn, sm, sk, tan, tak, tbk, tbm, scale, zero_point);
                    break;
                }
                case 2: {
                    matmul_quant_kernel<2, half><<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, sn, sm, sk, tan, tak, tbk, tbm, scale, zero_point);
                    break;
                }
                case 1: {
                    matmul_quant_kernel<1, half><<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, sn, sm, sk, tan, tak, tbk, tbm, scale, zero_point);
                    break;
                }
                default: {
                    TORCH_CHECK(false, "Unsupported number of quantization bits '", bits, "'");
                }
            }

            return c;
        }
        case at::kFloat: {
            float* b_ptr = b.data_ptr<float>();

            switch (bits) {
                case 8: {
                    matmul_quant_kernel<8, float><<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, sn, sm, sk, tan, tak, tbk, tbm, scale, zero_point);
                    break;
                }
                case 4: {
                    matmul_quant_kernel<4, float><<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, sn, sm, sk, tan, tak, tbk, tbm, scale, zero_point);
                    break;
                }
                case 2: {
                    matmul_quant_kernel<2, float><<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, sn, sm, sk, tan, tak, tbk, tbm, scale, zero_point);
                    break;
                }
                case 1: {
                    matmul_quant_kernel<1, float><<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, sn, sm, sk, tan, tak, tbk, tbm, scale, zero_point);
                    break;
                }
                default: {
                    TORCH_CHECK(false, "Unsupported number of quantization bits '", bits, "'");
                }
            }

            return c;
        }
        default: {
            TORCH_CHECK(false, "Unsupported type for comparison tensor '", typeb, "'");
        }
    }
}
