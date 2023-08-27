#include <torch/torch.h>
#include <immintrin.h>
#include <ATen/ParallelOpenMP.h>

using namespace torch;

#ifdef __AVX2__

const int64_t BLOCK_SIZE = 16; // = 128 bit / 8 bit
inline float_t dot_qint8_float32_cpu(int8_t* a, float_t* b, int64_t n, int64_t ta, int64_t tb, double scale, int64_t zero_point) {
  float_t sum = 0.0;
  int8_t* ai = a;
  float_t* bi = b;
  __m128i zero_i8 = _mm_set1_epi8(zero_point);
  for (int64_t i = 0; i < n / BLOCK_SIZE; i++) {
    __m128i a_vec_i8 = _mm_loadu_si128((__m128i_u*)ai); // a
    __m512 b_vec_f32 = _mm512_loadu_ps(bi); // b
    __m128i a_sub_i8 = _mm_sub_epi8(a_vec_i8, zero_i8); // a - zero_point
    __m512i a_sub_i32 = _mm512_cvtepi8_epi32(a_sub_i8); // a - zero_point (unpack to 32 bit)
    __m512 a_sub_f32 = _mm512_cvtepi32_ps(a_sub_i32); // float(a - zero_point)
    __m512 c_vec_f32 = _mm512_mul_ps(a_sub_f32, b_vec_f32); // float(a - zero_point) * b
    sum += _mm512_reduce_add_ps(c_vec_f32);
    ai += BLOCK_SIZE * ta;
    bi += BLOCK_SIZE * tb;
  }
  return scale*sum;
}

#else // __AVX2__

inline float_t dot_qint8_float32_cpu(int8_t* a, float_t* b, int64_t n, int64_t ta, int64_t tb, double scale, int64_t zero_point) {
  float_t vala;
  float_t valb;
  float_t sum = 0.0;
  int8_t* ai = a;
  float_t* bi = b;
  for (int64_t i = 0; i < n; i++) {
    vala = (float)((*ai) - zero_point);
    valb = (*bi);
    sum += vala * valb;
    ai += ta;
    bi += tb;
  }
  return scale*sum;
}

#endif // __AVX2__

Tensor matmul_qint8_float32_cpu(Tensor a, Tensor b) {
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

  Tensor c = torch::empty({sn, sm}, at::device(kCPU).dtype(torch::kFloat));

  int8_t* a_ptr = a.data_ptr<int8_t>();
  float_t* b_ptr = b.data_ptr<float_t>();
  float_t* c_ptr = c.data_ptr<float_t>();

  at::parallel_for(0, sn, 0, [&](int64_t i0, int64_t i1) {
    int8_t* posa;
    float_t* posb;
    for (int64_t i = i0; i < i1; i++) {
      for (int64_t j = 0; j < sm; j++) {
        posa = a_ptr + i * tan;
        posb = b_ptr + j * tbm;
        c_ptr[i * sm + j] = dot_qint8_float32_cpu(posa, posb, sk, tak, tbk, scale, zero_point);
      }
    }
  });

  return c;
}
