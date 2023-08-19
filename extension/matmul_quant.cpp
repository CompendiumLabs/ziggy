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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_quant_float", &matmul_quant_float, "Quantized Matmul");
}
