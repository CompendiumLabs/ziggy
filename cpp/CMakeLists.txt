cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(test LANGUAGES C CXX CUDA VERSION 1.0)

find_package(Torch REQUIRED)

add_executable(testing_cpu testing_cpu.cpp matmul_quant_cpu.cpp)
target_link_options(testing_cpu PUBLIC -lgomp)
target_link_libraries(testing_cpu "${TORCH_LIBRARIES}")

add_executable(testing_cuda testing_cuda.cpp matmul_quant_cuda.cu)
target_link_options(testing_cuda PUBLIC -lgomp)
target_link_libraries(testing_cuda "${TORCH_LIBRARIES}")

add_executable(simple simple.cpp simple.cu)
target_link_libraries(simple "${TORCH_LIBRARIES}")

set_target_properties(testing_cpu testing_cuda simple PROPERTIES CXX_STANDARD 14)
set_target_properties(testing_cuda simple PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

EXEC_PROGRAM(cat ARGS "/proc/cpuinfo" OUTPUT_VARIABLE CPUINFO)
STRING(FIND "${CPUINFO}" "avx512f" AVX512F_POS)
IF(AVX512F_POS EQUAL -1)
  MESSAGE(STATUS "AVX512F not supported")
ELSE()
  MESSAGE(STATUS "AVX512F supported")
  target_compile_options(testing_cpu PUBLIC -fopenmp -mavx512f)
ENDIF()
