#include <torch/extension.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/Dispatch.h>

#include <iostream>

using namespace torch;
using namespace at::native::CPU_CAPABILITY;

Tensor matmul_quant_float(Tensor a, Tensor b) {
  at::ScalarType typea = a.scalar_type();
  // at::ScalarType typeb = b.scalar_type();

  at::IntArrayRef sizesa = a.sizes();
  at::IntArrayRef sizesb = b.sizes();
  assert(sizesa[1] == sizesb[0]);

  at::IntArrayRef stridesa = a.strides();
  at::IntArrayRef stridesb = b.strides();

  int64_t sizesc[] = {sizesa[0], sizesb[1]};
  Tensor c = torch::empty(sizesc, at::device(kCPU).dtype(torch::kFloat));

  double scale = a.q_scale();
  int64_t zero_point = a.q_zero_point();

  AT_DISPATCH_QINT_TYPES(typea, "matmul_quant_float", [&]() {
    underlying_t* a_ptr = a.data_ptr<underlying_t>();
    float_t* b_ptr = b.data_ptr<float_t>();
    float_t* c_ptr = c.data_ptr<float_t>();

    int64_t idxa, idxb;
    float_t vala, valb;
    float_t sum;

    for (int64_t i = 0; i < sizesa[0]; i++) {
      for (int64_t j = 0; j < sizesb[1]; j++) {
        sum = 0.0;
        for (int64_t k = 0; k < sizesa[1]; k++) {
          idxa = i * stridesa[0] + k * stridesa[1];
          idxb = k * stridesb[0] + j * stridesb[1];
          vala = scale * (a_ptr[idxa] - zero_point);
          valb = b_ptr[idxb];
          sum += vala * valb;
        }
        c_ptr[i * sizesc[1] + j] = sum;
      }
    }
  });

  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_quant_float", &matmul_quant_float, "Quantized Matmul");
}
