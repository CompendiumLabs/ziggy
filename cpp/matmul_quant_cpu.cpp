#include <torch/torch.h>
#include <immintrin.h>
#include <ATen/ParallelOpenMP.h>

using namespace torch;

#ifdef __AVX2__

// sub-8 quantization is tricky with AVX2, but memory pressure is lower on CPU

constexpr int64_t BLOCK_SIZE = 16; // = 128 bit / 8 bit

inline float dot_quant_float_avx2(uint8_t* a, float* b, int64_t n, int64_t ta, int64_t tb, float scale, float zero_point) {
  float sum = 0.0;
  uint8_t* ai = a;
  float* bi = b;
  __m512 zero_f32 = _mm512_set1_ps(zero_point);
  for (int64_t i = 0; i < n / BLOCK_SIZE; i++) {
    __m128i a_vec_i8 = _mm_loadu_si128((__m128i*)ai); // a
    __m512 b_vec_f32 = _mm512_loadu_ps(bi); // b
    __m512i a_vec_i32 = _mm512_cvtepu8_epi32(a_vec_i8); // int(a)
    __m512 a_vec_f32 = _mm512_cvtepi32_ps(a_vec_i32); // float(a)
    __m512 a_sub_f32 = _mm512_sub_ps(a_vec_f32, zero_f32); // float(a) - zero_point
    __m512 c_vec_f32 = _mm512_mul_ps(a_sub_f32, b_vec_f32); // float(a - zero_point) * b
    sum += _mm512_reduce_add_ps(c_vec_f32);
    ai += BLOCK_SIZE * ta;
    bi += BLOCK_SIZE * tb;
  }
  return scale*sum;
}

#endif // __AVX2__

template <unsigned int bits>
inline float dot_quant_float_cpu(uint8_t* a, float* b, int64_t sk, int64_t tak, int64_t tbk, float scale, float zero_point) {
  constexpr uint8_t mask = (1 << bits) - 1;

  uint8_t vala_i;
  float vala_f;
  float valb_f;

  uint8_t* posa = a;
  float* posb = b;
  float sum = 0.0;

  for (int64_t k = 0; k < sk; k++) {
    vala_i = (*posa);

    for (int s = 0; s < 8; s += bits) {
      vala_i = (vala_i >> s) & mask;
      vala_f = (float)vala_i - zero_point;
      valb_f = (*posb);
      sum += vala_f * valb_f;
      posb += tbk;
    }

    posa += tak;
  }

  return scale*sum;
}

template <unsigned int bits>
inline void quant_pack_float_cpu(float* a, uint8_t* b, int64_t sk, int64_t tak, int64_t tbk, float scale, float zero_point) {
  constexpr int qFact = 8 / bits;
  constexpr int pMax = (1 << bits) - 1;

  const int64_t sk_p = sk / qFact;
  const float pMax_f = (float)pMax;

  float vala_f;
  uint8_t vala_i;
  uint8_t valb;

  float* posa = a;
  uint8_t* posb = b;

  for (int64_t k = 0; k < sk_p; k++) {
    valb = 0;

    for (int s = 0; s < 8; s += bits) {
      vala_f = (*posa);
      vala_f = vala_f / scale + zero_point;
      vala_f = std::max(0.0f, std::min(vala_f, pMax_f));
      vala_i = (uint8_t)vala_f;
      valb |= vala_i << s;
      posa += tak;
    }

    (*posb) = valb;
    posb += tbk;
  }
}

Tensor matmul_quant_cpu(Tensor a, Tensor b, unsigned int bits, float scale, float zero_point) {
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
  assert(typeb == torch::kFloat);
  assert((8 / bits) * sizesa[1] == sizesb[0]);

  int64_t sn = sizesa[0];
  int64_t sm = sizesb[1];
  int64_t sk = sizesa[1];
  int64_t tan = stridesa[0];
  int64_t tak = stridesa[1];
  int64_t tbk = stridesb[0];
  int64_t tbm = stridesb[1];

  Tensor c = torch::empty({sn, sm}, at::device(kCPU).dtype(torch::kFloat));

  uint8_t* a_ptr = a.data_ptr<uint8_t>();
  float* b_ptr = b.data_ptr<float>();
  float* c_ptr = c.data_ptr<float>();

  at::parallel_for(0, sn, 0, [&](int64_t i0, int64_t i1) {
    uint8_t* posa;
    float* posb;
    float res;
    for (int64_t i = i0; i < i1; i++) {
      for (int64_t j = 0; j < sm; j++) {
        posa = a_ptr + i * tan;
        posb = b_ptr + j * tbm;

        switch (bits) {
          case 8: {
            #ifdef __AVX2__
            res = dot_quant_float_avx2(posa, posb, sk, tak, tbk, scale, zero_point);
            #else // __AVX2__
            res = dot_quant_float_cpu<8>(posa, posb, sk, tak, tbk, scale, zero_point);
            #endif // __AVX2__
            break;
          }
          case 4: {
            res = dot_quant_float_cpu<4>(posa, posb, sk, tak, tbk, scale, zero_point);
            break;
          }
          case 2: {
            res = dot_quant_float_cpu<2>(posa, posb, sk, tak, tbk, scale, zero_point);
            break;
          }
          case 1: {
            res = dot_quant_float_cpu<1>(posa, posb, sk, tak, tbk, scale, zero_point);
            break;
          }
          default: {
            TORCH_CHECK(false, "Unsupported number of quantization bits '", bits, "'");
          }
        }

        c_ptr[i * sm + j] = res;
      }
    }
  });

  return c;
}

Tensor quantize_and_pack_cpu(Tensor a, unsigned int bits, float scale, float zero_point) {
  at::ScalarType typea = a.scalar_type();
  at::IntArrayRef sizesa = a.sizes();
  at::IntArrayRef stridesa = a.strides();

  assert(typea == torch::kFloat);

  int64_t sn = sizesa[0];
  int64_t sk = sizesa[1];
  int64_t tan = stridesa[0];
  int64_t tak = stridesa[1];

  int64_t sk_p = sk / (8 / bits);
  int64_t tbm = sk_p;
  int64_t tbk = 1;

  Tensor b = torch::empty({sn, sk_p}, torch::device(kCPU).dtype(torch::kUInt8));

  float* a_ptr = a.data_ptr<float>();
  uint8_t* b_ptr = b.data_ptr<uint8_t>();

  at::parallel_for(0, sn, 0, [&](int64_t i0, int64_t i1) {
    float* posa;
    uint8_t* posb;
    for (int64_t i = i0; i < i1; i++) {
        posa = a_ptr + i * tan;
        posb = b_ptr + i * tbm;

        switch (bits) {
          case 8: {
            quant_pack_float_cpu<8>(posa, posb, sk, tak, tbk, scale, zero_point);
            break;
          }
          case 4: {
            quant_pack_float_cpu<4>(posa, posb, sk, tak, tbk, scale, zero_point);
            break;
          }
          case 2: {
            quant_pack_float_cpu<2>(posa, posb, sk, tak, tbk, scale, zero_point);
            break;
          }
          case 1: {
            quant_pack_float_cpu<1>(posa, posb, sk, tak, tbk, scale, zero_point);
            break;
          }
          default: {
            TORCH_CHECK(false, "Unsupported number of quantization bits '", bits, "'");
          }
        }
    }
  });

  return b;
}
