#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <torch/torch.h>

using namespace torch;

constexpr unsigned int kWarpSize = 32;

__global__ void matmul_qint8_float16_kernel(int8_t* a, __half* b, __half* c, int64_t sn, int64_t sm, int64_t sk, int64_t tan, int64_t tak, int64_t tbk, int64_t tbm, double scale, int64_t zero_point) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < sn && j < sm) {
    __half sum = __float2half(0.0f);
    int8_t* posa = a + i * tan;
    __half* posb = b + j * tbm;

    for (int64_t k = 0; k < sk; k++) {
      int8_t vala = (*posa) - zero_point;
      __half valb = (*posb);

      __half vala_half = __float2half((float)vala);
      sum = __hadd(sum, __hmul(vala_half, valb));

      posa += tak;
      posb += tbk;
    }

    c[i * sm + j] = __hmul(__double2half(scale), sum);
  }
}

Tensor matmul_qint8_float16_cuda(Tensor a, Tensor b) {
  at::ScalarType typea = a.scalar_type();
  at::ScalarType typeb = b.scalar_type();
  double scale = a.q_scale();
  int64_t zero_point = a.q_zero_point();

  at::IntArrayRef sizesa = a.sizes();
  at::IntArrayRef sizesb = b.sizes();
  at::IntArrayRef stridesa = a.strides();
  at::IntArrayRef stridesb = b.strides();

  assert(typea == at::kQInt8);
  assert(typeb == at::kHalf);
  assert(sizesa[1] == sizesb[0]);

  int64_t sn = sizesa[0];
  int64_t sm = sizesb[1];
  int64_t sk = sizesa[1];
  int64_t tan = stridesa[0];
  int64_t tak = stridesa[1];
  int64_t tbk = stridesb[0];
  int64_t tbm = stridesb[1];

  Tensor c = torch::empty({sn, sm}, at::device(kCUDA).dtype(torch::kHalf));

  int8_t* a_ptr = a.data_ptr<int8_t>();
  at::Half* b_ptr0 = b.data_ptr<at::Half>();
  at::Half* c_ptr0 = c.data_ptr<at::Half>();
  __half* b_ptr = reinterpret_cast<__half*>(b_ptr0);
  __half* c_ptr = reinterpret_cast<__half*>(c_ptr0);

  dim3 threads(kWarpSize, kWarpSize);
  dim3 blocks((sn + threads.x - 1) / threads.x, (sm + threads.y - 1) / threads.y);
  matmul_qint8_float16_kernel<<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, sn, sm, sk, tan, tak, tbk, tbm, scale, zero_point);

  return c;
}
