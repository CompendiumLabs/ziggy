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
  const int sk_packed = sk / qFact;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < sn && j < sm) {
    packed_t* posa = a + i * tan / qFact;
    __half* posb = b + j * tbm;
    __half sum = __float2half(0.0f);

    for (int k = 0; k < sk_packed; k++) {
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
  const int sk_packed = sk / qFact;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < sn && j < sm) {
    packed_t* posa = a + i * tan / qFact;
    float* posb = b + j * tbm;
    float sum = 0.0;

    for (int k = 0; k < sk_packed; k++) {
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

// we need to use unsigned intN packing here
template <unsigned int bits>
__global__ void quantize_and_pack_float(float* a, uint8_t* b, int64_t sn, int64_t sk, int64_t tan, int64_t tak, double scale, int64_t zero_point) {
  constexpr int qFact = 8 / bits;
  constexpr int pMax = (1 << (bits - 1)) - 1;
  constexpr float pMax_f = (float)pMax;

  const int sk_packed = sk / qFact;
  const float scale_f = (float)scale;
  const float zero_point_f = (float)zero_point;

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < sn) {
    float* posa = a + i * tan;

    for (int k = 0; k < sk_packed; k++) {
      // init zero bits
      uint8_t valb = 0;

      // pack into packed_t-sized values
      for (int s = 0; s < 8; s += bits) {
        float vala = (*posa);
        vala = vala / scale_f + zero_point_f;
        vala = max(-pMax_f, min(pMax_f, vala)) + pMax_f;
        uint8_t vala_i = __float2uint_rn(vala);
        valb |= vala_i << s;
        posa += tak;
      }

      // store packed value
      b[i * sk_packed + k] = valb;
    }
  }
}

template <unsigned int bits>
__global__ void quantize_and_pack_half(__half* a, uint8_t* b, int64_t sn, int64_t sk, int64_t tan, int64_t tak, double scale, int64_t zero_point) {
  constexpr int qFact = 8 / bits;
  constexpr int pMax = (1 << (bits - 1)) - 1;

  const int sk_packed = sk / qFact;
  const __half pMax_h = (__half)pMax;
  const __half scale_h = __double2half(scale);
  const __half zerop_h = __int2half_rn(zero_point);

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < sn) {
    __half* posa = a + i * tan;

    for (int k = 0; k < sk_packed; k++) {
      // init zero bits
      uint8_t valb = 0;

      // pack into packed_t-sized values
      for (int s = 0; s < 8; s += bits) {
        __half vala = (*posa);
        vala = vala / scale_h + zerop_h;
        vala = __hmax(-pMax_h, __hmin(pMax_h, vala)) + pMax_h;
        uint8_t vala_i = __half2int_rn(vala);
        valb |= vala_i << s;
        posa += tak;
      }

      // store packed value
      b[i * sk_packed + k] = valb;
    }
  }
}

Tensor matmul_qint8_cuda(Tensor a, Tensor b) {
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

  int8_t* a_ptr = a.data_ptr<int8_t>();

  dim3 threads(kWarpSize, kWarpSize);
  dim3 blocks((sn + threads.x - 1) / threads.x, (sm + threads.y - 1) / threads.y);

  switch (b.scalar_type()) {
    case torch::ScalarType::Half: {
      Tensor c = torch::empty({sn, sm}, at::device(kCUDA).dtype(torch::kHalf));

      __half* b_ptr = reinterpret_cast<__half*>(b.data_ptr<at::Half>());
      __half* c_ptr = reinterpret_cast<__half*>(c.data_ptr<at::Half>());

      matmul_quant_half_kernel<8, int8_t><<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, sn, sm, sk, tan, tak, tbk, tbm, scale, zero_point);

      return c;
    }
    case torch::ScalarType::Float: {
      Tensor c = torch::empty({sn, sm}, at::device(kCUDA).dtype(torch::kFloat));

      float* b_ptr = b.data_ptr<float>();
      float* c_ptr = c.data_ptr<float>();

      matmul_quant_float_kernel<8, int8_t><<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, sn, sm, sk, tan, tak, tbk, tbm, scale, zero_point);

      return c;
    }
    default: {
      throw std::runtime_error("Unsupported type for comparison tensors");
    }
  }
}

Tensor quantize_and_pack(Tensor a, unsigned int bits, double scale, int64_t zero_point) {
  at::IntArrayRef sizesa = a.sizes();
  at::IntArrayRef stridesa = a.strides();

  int64_t sn = sizesa[0];
  int64_t sk = sizesa[1];
  int64_t tan = stridesa[0];
  int64_t tak = stridesa[1];

  dim3 threads(kWarpSize);
  dim3 blocks((sn + threads.x - 1) / threads.x);

  int64_t sk_packed = sk / (8 / bits);
  Tensor b = torch::empty({sn, sk_packed}, at::device(kCUDA).dtype(torch::kUInt8));

  switch (a.scalar_type()) {
    case torch::ScalarType::Float: {
      float* a_ptr = a.data_ptr<float>();
      uint8_t* b_ptr = b.data_ptr<uint8_t>();

      switch (bits) {
        case 8:
          quantize_and_pack_float<8><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, tan, tak, scale, zero_point);
          break;
        case 4:
          quantize_and_pack_float<4><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, tan, tak, scale, zero_point);
          break;
        default:
          throw std::runtime_error("Unsupported number of quantization bits");
      }

      break;
    }
    case torch::ScalarType::Half: {
      __half* a_ptr = reinterpret_cast<__half*>(a.data_ptr<at::Half>());
      uint8_t* b_ptr = b.data_ptr<uint8_t>();

      switch (bits) {
        case 8:
          quantize_and_pack_half<8><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, tan, tak, scale, zero_point);
          break;
        case 4:
          quantize_and_pack_half<4><<<blocks, threads>>>(a_ptr, b_ptr, sn, sk, tan, tak, scale, zero_point);
          break;
        default:
          throw std::runtime_error("Unsupported number of quantization bits");
      }

      break;
    }
    default: {
      throw std::runtime_error("Unsupported type for comparison tensors");
    }
  }

  return b;
}
