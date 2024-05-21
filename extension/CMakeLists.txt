cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(test LANGUAGES C CXX CUDA VERSION 1.0)

option(ZIGGY_AVX512 "Enable AVX512" ON)

find_package(Torch REQUIRED)

add_executable(testing_cpu testing_cpu.cpp matmul_quant_cpu.cpp)
target_link_options(testing_cpu PUBLIC -lgomp)
target_link_libraries(testing_cpu "${TORCH_LIBRARIES}")

add_executable(testing_cuda testing_cuda.cpp matmul_quant_cuda.cu)
target_link_options(testing_cuda PUBLIC -lgomp)
target_link_libraries(testing_cuda "${TORCH_LIBRARIES}")

set_target_properties(testing_cpu testing_cuda PROPERTIES CXX_STANDARD 17)
target_compile_options(testing_cpu PUBLIC -fopenmp)

if(ZIGGY_AVX512)
  exec_program(cat ARGS "/proc/cpuinfo" OUTPUT_VARIABLE CPUINFO)
  string(FIND "${CPUINFO}" "avx512f" AVX512F_POS)
  if(AVX512F_POS EQUAL -1)
    message(STATUS "AVX512F not supported")
  else()
    message(STATUS "AVX512F supported")
    target_compile_options(testing_cpu PUBLIC -mavx512f)
  endif()
else()
  message(STATUS "AVX512F disabled")
endif()
