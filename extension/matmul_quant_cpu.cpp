#include <math.h>
#include <torch/torch.h>
#include <immintrin.h>
#include <ATen/ParallelOpenMP.h>

using namespace torch;

template <unsigned int bits>
inline void quantize_float_cpu(float* a, uint8_t* b, int64_t sk, int64_t tak, float scale, float zero_point) {
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
            vala_f = std::max(0.0f, std::min(pMax_f, vala_f));
            vala_i = (uint8_t)round(vala_f);
            valb |= vala_i << s;
            posa += tak;
        }

        (*posb) = valb;
        posb++;
    }
}

template <unsigned int bits>
inline void dequantize_float_cpu(uint8_t* a, float* b, int64_t sk, float scale, float zero_point) {
    constexpr int qFact = 8 / bits;

    const int64_t sk_p = sk / qFact;
    const uint8_t pMax = (1 << bits) - 1;

    uint8_t vala;
    float valb;

    uint8_t* posa = a;
    float* posb = b;

    for (int64_t k = 0; k < sk_p; k++) {
        vala = (*posa);

        for (int s = 0; s < 8; s += bits) {
            valb = (float)((vala >> s) & pMax);
            (*posb) = scale * (valb - zero_point);
            posb++;
        }

        posa++;
    }
}

#ifdef __AVX2__

constexpr int BLOCK_SIZE_FLOAT = 16; // = 512 bit / 32 bit

inline float dot_float_avx2(float* a, float* b, int64_t n, int64_t tb) {
    __m512 sum = _mm512_setzero_ps();
    __m512 vala;
    __m512 valb;
    float* posa = a;
    float* posb = b;
    for (int64_t i = 0; i < n; i += BLOCK_SIZE_FLOAT) {
        vala = _mm512_loadu_ps(posa);
        valb = _mm512_loadu_ps(posb);
        sum = _mm512_fmadd_ps(vala, valb, sum);
        posa += BLOCK_SIZE_FLOAT;
        posb += tb * BLOCK_SIZE_FLOAT;
    }
    return _mm512_reduce_add_ps(sum);
}

#endif

inline float dot_float_cpu(float* a, float* b, int64_t n, int64_t tb) {
    float vala;
    float valb;
    float sum = 0.0;
    float* posa = a;
    float* posb = b;
    for (int64_t i = 0; i < n; i++) {
        vala = (*posa);
        valb = (*posb);
        sum += vala * valb;
        posa++;
        posb += tb;
    }
    return sum;
}

#ifdef __AVX2__

// sub-8 quantization is tricky with AVX2, but memory pressure is lower on CPU

constexpr int64_t BLOCK_SIZE_QUANT = 16; // = 128 bit / 8 bit

inline float dot_quant_avx2(uint8_t* a, float* b, int64_t n, int64_t tb, float scale, float zero_point) {
    float sum = 0.0;
    uint8_t* ai = a;
    float* bi = b;
    __m512 zero_f32 = _mm512_set1_ps(zero_point);
    for (int64_t i = 0; i < n; i += BLOCK_SIZE_QUANT) {
        __m128i a_vec_i8 = _mm_loadu_si128((__m128i*)ai); // a
        __m512 b_vec_f32 = _mm512_loadu_ps(bi); // b
        __m512i a_vec_i32 = _mm512_cvtepu8_epi32(a_vec_i8); // int(a)
        __m512 a_vec_f32 = _mm512_cvtepi32_ps(a_vec_i32); // float(a)
        __m512 a_sub_f32 = _mm512_sub_ps(a_vec_f32, zero_f32); // float(a) - zero_point
        __m512 c_vec_f32 = _mm512_mul_ps(a_sub_f32, b_vec_f32); // float(a - zero_point) * b
        sum += _mm512_reduce_add_ps(c_vec_f32);
        ai += BLOCK_SIZE_QUANT;
        bi += BLOCK_SIZE_QUANT * tb;
    }
    return scale*sum;
}

#endif // __AVX2__

template <unsigned int bits>
inline float dot_quant_cpu(uint8_t* a, float* b, int64_t n, int64_t tb, float scale, float zero_point) {
    constexpr int qFact = 8 / bits;
    constexpr uint8_t pMax = (1 << bits) - 1;

    const int64_t n_p = n / qFact;

    uint8_t vala_i;
    float vala_f;
    float valb_f;

    uint8_t* posa = a;
    float* posb = b;
    float sum = 0.0;

    for (int64_t k = 0; k < n_p; k++) {
        vala_i = (*posa);

        for (int s = 0; s < 8; s += bits) {
            vala_f = (float)((vala_i >> s) & pMax);
            valb_f = (*posb);
            sum += (vala_f - zero_point) * valb_f;
            posb += tb;
        }

        posa++;
    }

    return scale*sum;
}

Tensor quantize_cpu(Tensor a, unsigned int bits, float scale, float zero_point) {
    at::ScalarType typea = a.scalar_type();
    at::IntArrayRef sizesa = a.sizes();
    at::IntArrayRef stridesa = a.strides();

    int64_t sn = sizesa[0];
    int64_t sk = sizesa[1];
    int64_t tan = stridesa[0];
    int64_t tak = stridesa[1];
    int64_t sk_p = sk / (8 / bits);

    assert(typea == torch::kFloat);

    Tensor b = torch::empty({sn, sk_p}, torch::device(kCPU).dtype(torch::kUInt8));
    float* a_ptr = a.data_ptr<float>();
    uint8_t* b_ptr = b.data_ptr<uint8_t>();

    at::parallel_for(0, sn, 0, [&](int64_t i0, int64_t i1) {
        float* posa;
        uint8_t* posb;
        for (int64_t i = i0; i < i1; i++) {
            posa = a_ptr + i * tan;
            posb = b_ptr + i * sk_p;

            switch (bits) {
                case 8: {
                    quantize_float_cpu<8>(posa, posb, sk, tak, scale, zero_point);
                    break;
                }
                case 4: {
                    quantize_float_cpu<4>(posa, posb, sk, tak, scale, zero_point);
                    break;
                }
                case 2: {
                    quantize_float_cpu<2>(posa, posb, sk, tak, scale, zero_point);
                    break;
                }
                case 1: {
                    quantize_float_cpu<1>(posa, posb, sk, tak, scale, zero_point);
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

// this assumes normal strides since `a` is packed anyway
Tensor dequantize_cpu(Tensor a, unsigned int bits, float scale, float zero_point, at::ScalarType typeb) {
    at::ScalarType typea = a.scalar_type();
    at::IntArrayRef sizesa = a.sizes();
    at::IntArrayRef stridesa = a.strides();

    int64_t sn = sizesa[0];
    int64_t sk_p = sizesa[1];
    int64_t sk = sk_p * (8 / bits);

    assert(typea == torch::kUInt8);
    assert(typeb == torch::kFloat);

    Tensor b = torch::empty({sn, sk}, torch::device(kCPU).dtype(typeb));
    uint8_t* a_ptr = a.data_ptr<uint8_t>();
    float* b_ptr = b.data_ptr<float>();

    at::parallel_for(0, sn, 0, [&](int64_t i0, int64_t i1) {
        uint8_t* posa;
        float* posb;

        for (int64_t i = i0; i < i1; i++) {
            posa = a_ptr + i * sk_p;
            posb = b_ptr + i * sk;

            switch (bits) {
                case 8: {
                    dequantize_float_cpu<8>(posa, posb, sk, scale, zero_point);
                    break;
                }
                case 4: {
                    dequantize_float_cpu<4>(posa, posb, sk, scale, zero_point);
                    break;
                }
                case 2: {
                    dequantize_float_cpu<2>(posa, posb, sk, scale, zero_point);
                    break;
                }
                case 1: {
                    dequantize_float_cpu<1>(posa, posb, sk, scale, zero_point);
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

Tensor matmul_float_cpu(Tensor a, Tensor b) {
    at::Device devicea = a.device();
    at::Device deviceb = b.device();

    at::ScalarType typea = a.scalar_type();
    at::ScalarType typeb = b.scalar_type();

    at::IntArrayRef sizesa = a.sizes();
    at::IntArrayRef sizesb = b.sizes();
    at::IntArrayRef stridesa = a.strides();
    at::IntArrayRef stridesb = b.strides();

    int64_t sn = sizesa[0];
    int64_t sk = sizesa[1];
    int64_t sm = sizesb[1];
    int64_t tan = stridesa[0];
    int64_t tbk = stridesb[0];
    int64_t tbm = stridesb[1];

    assert(devicea == deviceb);
    assert(typea == torch::kFloat);
    assert(typeb == torch::kFloat);

    Tensor c = torch::empty({sn, sm}, at::device(kCPU).dtype(torch::kFloat));
    float* a_ptr = a.data_ptr<float>();
    float* b_ptr = b.data_ptr<float>();
    float* c_ptr = c.data_ptr<float>();

    at::parallel_for(0, sn, 0, [&](int64_t i0, int64_t i1) {
        float* posa;
        float* posb;
        float res;
        for (int64_t i = i0; i < i1; i++) {
            for (int64_t j = 0; j < sm; j++) {
                posa = a_ptr + i * sk;
                posb = b_ptr + j * tbm;
                #ifdef __AVX2__
                if (sk % BLOCK_SIZE_FLOAT == 0) {
                    res = dot_float_avx2(posa, posb, sk, tbk);
                } else {
                    res = dot_float_cpu(posa, posb, sk, tbk);
                }
                #else
                res = dot_float_cpu(posa, posb, sk, tbk);
                #endif
                c_ptr[i * sm + j] = res;
            }
        }
    });

    return c;
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

    int64_t sn = sizesa[0];
    int64_t sk_p = sizesa[1];
    int64_t sk = sizesb[0];
    int64_t sm = sizesb[1];
    int64_t tan = stridesa[0];
    int64_t tbk = stridesb[0];
    int64_t tbm = stridesb[1];

    assert(devicea == deviceb);
    assert(typea == torch::kUInt8);
    assert(typeb == torch::kFloat);
    assert((8 / bits) * sk_p == sk);

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
                posa = a_ptr + i * sk_p;
                posb = b_ptr + j * tbm;

                switch (bits) {
                    case 8: {
                        #ifdef __AVX2__
                        if (sk % BLOCK_SIZE_QUANT == 0) {
                            res = dot_quant_avx2(posa, posb, sk, tbk, scale, zero_point);
                        } else {
                            res = dot_quant_cpu<8>(posa, posb, sk, tbk, scale, zero_point);
                        }
                        #else // __AVX2__
                        res = dot_quant_cpu<8>(posa, posb, sk, tbk, scale, zero_point);
                        #endif // __AVX2__
                        break;
                    }
                    case 4: {
                        res = dot_quant_cpu<4>(posa, posb, sk, tbk, scale, zero_point);
                        break;
                    }
                    case 2: {
                        res = dot_quant_cpu<2>(posa, posb, sk, tbk, scale, zero_point);
                        break;
                    }
                    case 1: {
                        res = dot_quant_cpu<1>(posa, posb, sk, tbk, scale, zero_point);
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
