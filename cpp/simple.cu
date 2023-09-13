#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <torch/torch.h>

using namespace torch;

constexpr unsigned int kWarpSize = 32;

__global__ void pack_float(float* a, uint8_t* b, int64_t sn, double scale, int64_t zero_point) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr unsigned int bits = 4;
    constexpr unsigned int pMax = (1 << (bits - 1)) - 1;
    constexpr float pMaxf = (float)pMax;

    if (i < sn) {
        float va = a[i];
        va = va / scale + zero_point;
        va = fmaxf(-pMaxf, fminf(pMaxf, va));
        uint8_t vb = __float2uint_rn(va);
        b[i] = vb;
    }
}

Tensor pack(Tensor a, double scale, int64_t zero_point) {
    at::IntArrayRef sizes = a.sizes();
    int64_t sn = sizes[0];
    Tensor b = torch::empty({sn}, at::device(kCUDA).dtype(torch::kUInt8));

    float* a_ptr = a.data_ptr<float>();
    uint8_t* b_ptr = b.data_ptr<uint8_t>();

    dim3 threads(kWarpSize);
    dim3 blocks((sn + threads.x - 1) / threads.x);
    pack_float<<<blocks, threads>>>(a_ptr, b_ptr, sn, scale, zero_point);

    return b;
}
