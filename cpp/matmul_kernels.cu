#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <torch/torch.h>

using namespace torch;

constexpr unsigned int kWarpSize = 32;

template <unsigned int bits, typename packed_t>
__global__ void matmul_quant_half_kernel(packed_t* a, __half* b, __half* c, int64_t sn, int64_t sm, int64_t sk, int64_t tan, int64_t tak, int64_t tbk, int64_t tbm, double scale, int64_t zero_point) {
  constexpr int pSize = 8 * sizeof(packed_t);
  constexpr int qFact = pSize / bits;
  constexpr packed_t mask = (1 << bits) - 1;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < sn && j < sm) {
    packed_t* posa = a + i * tan / qFact;
    __half* posb = b + j * tbm;
    __half sum = __float2half(0.0f);

    for (int k = 0; k < sk / qFact; k++) {
      // load packed values
      packed_t vala = (*posa);

      // unpack into bits-sized values
      for (int s = 0; s < pSize; s += bits) {
        int8_t vala_i = (int8_t)((vala >> s) & mask);
        __half vala_h = __float2half((float)(vala_i - zero_point));
        __half valb_h = (*posb);
        sum = __hadd(sum, __hmul(vala_h, valb_h));
        posb += tbk;
      }

      // increment base value
      posa += tak;
    }

    c[i * sm + j] = __hmul(__double2half(scale), sum);
  }
}

template <unsigned int bits, typename packed_t>
__global__ void matmul_quant_float_kernel(packed_t* a, float* b, float* c, int64_t sn, int64_t sm, int64_t sk, int64_t tan, int64_t tak, int64_t tbk, int64_t tbm, double scale, int64_t zero_point) {
  constexpr int pSize = 8 * sizeof(packed_t);
  constexpr int qFact = pSize / bits;
  constexpr packed_t mask = (1 << bits) - 1;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < sn && j < sm) {
    packed_t* posa = a + i * tan / qFact;
    float* posb = b + j * tbm;
    float sum = 0.0;

    for (int k = 0; k < sk / qFact; k++) {
      // load packed values
      packed_t vala = (*posa);

      // unpack into bits-sized values
      for (int s = 0; s < pSize; s += bits) {
        int8_t vala_i = (int8_t)((vala >> s) & mask);
        float vala_f = (float)(vala_i - zero_point);
        float valb_f = (*posb);
        sum += vala_f * valb_f;
        posb += tbk;
      }

      // increment base value
      posa += tak;
    }

    c[i * sm + j] = scale*sum;
  }
}

Tensor matmul_qint8_half_cuda(Tensor a, Tensor b) {
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
  matmul_quant_half_kernel<8, int8_t><<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, sn, sm, sk, tan, tak, tbk, tbm, scale, zero_point);

  return c;
}

Tensor matmul_qint8_float_cuda(Tensor a, Tensor b) {
  at::ScalarType typea = a.scalar_type();
  at::ScalarType typeb = b.scalar_type();
  double scale = a.q_scale();
  int64_t zero_point = a.q_zero_point();

  at::IntArrayRef sizesa = a.sizes();
  at::IntArrayRef sizesb = b.sizes();
  at::IntArrayRef stridesa = a.strides();
  at::IntArrayRef stridesb = b.strides();

  assert(typea == at::kQInt8);
  assert(typeb == at::kFloat);
  assert(sizesa[1] == sizesb[0]);

  int64_t sn = sizesa[0];
  int64_t sm = sizesb[1];
  int64_t sk = sizesa[1];
  int64_t tan = stridesa[0];
  int64_t tak = stridesa[1];
  int64_t tbk = stridesb[0];
  int64_t tbm = stridesb[1];

  Tensor c = torch::empty({sn, sm}, at::device(kCUDA).dtype(torch::kFloat));

  int8_t* a_ptr = a.data_ptr<int8_t>();
  float* b_ptr = b.data_ptr<float>();
  float* c_ptr = c.data_ptr<float>();

  dim3 threads(kWarpSize, kWarpSize);
  dim3 blocks((sn + threads.x - 1) / threads.x, (sm + threads.y - 1) / threads.y);
  matmul_quant_float_kernel<8, int8_t><<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, sn, sm, sk, tan, tak, tbk, tbm, scale, zero_point);

  return c;
}
