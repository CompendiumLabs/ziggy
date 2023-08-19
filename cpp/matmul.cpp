#include <iostream>

#include <torch/torch.h>
#include <ATen/ParallelOpenMP.h>
#include <ATen/Dispatch.h>

using namespace torch;

Tensor matmul_quant_float(Tensor a, Tensor b) {
  at::ScalarType typea = a.scalar_type();
  // at::ScalarType typeb = b.scalar_type();
  double scale = a.q_scale();
  int64_t zero_point = a.q_zero_point();

  at::IntArrayRef sizesa = a.sizes();
  at::IntArrayRef sizesb = b.sizes();
  at::IntArrayRef stridesa = a.strides();
  at::IntArrayRef stridesb = b.strides();
  assert(sizesa[1] == sizesb[0]);

  int64_t sn = sizesa[0];
  int64_t sm = sizesb[1];
  int64_t sk = sizesa[1];
  int64_t tan = stridesa[0];
  int64_t tak = stridesa[1];
  int64_t tbk = stridesb[0];
  int64_t tbm = stridesb[1];
  Tensor c = torch::empty({sn, sm}, at::device(kCPU).dtype(torch::kFloat));

  AT_DISPATCH_QINT_TYPES(typea, "matmul_quant_float", [&]() {
    underlying_t* a_ptr = a.data_ptr<underlying_t>();
    float_t* b_ptr = b.data_ptr<float_t>();
    float_t* c_ptr = c.data_ptr<float_t>();

    at::parallel_for(0, sn, 0, [&](int64_t i0, int64_t i1) {
      underlying_t* ptra;
      float_t* ptrb;
      float_t vala, valb;
      float_t sum;
      for (int64_t i = i0; i < i1; i++) {
        for (int64_t j = 0; j < sm; j++) {
          sum = 0.0;
          ptra = a_ptr + i * tan;
          ptrb = b_ptr + j * tbm;
          for (int64_t k = 0; k < sk; k++) {
            vala = scale * ((*ptra) - zero_point);
            valb = (*ptrb);
            sum += vala * valb;
            ptra += tak;
            ptrb += tbk;
          }
          c_ptr[i * sm + j] = sum;
        }
      }
    });
  });

  return c;
}

void test_quantized() {
  std::cout << torch::get_num_threads() << std::endl;
  std::cout << torch::get_num_interop_threads() << std::endl;

  at::IntArrayRef sizes = {4096, 256};
  Tensor a = torch::ones(sizes);
  Tensor b = torch::ones(sizes);

  Tensor qa = quantize_per_tensor(a, 2.0/128, 0, at::kQInt8);
  Tensor c = matmul_quant_float(qa.transpose(0, 1), b);

  std::cout << a.sizes() << " | " << a.mean() << std::endl;
  std::cout << b.sizes() << " | " << b.mean() << std::endl;
  std::cout << c.sizes() << " | " << c.mean() << std::endl;
}

int main() {
  test_quantized();
}
