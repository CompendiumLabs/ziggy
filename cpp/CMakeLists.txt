cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(test LANGUAGES C CXX CUDA VERSION 1.0)

find_package(Torch REQUIRED)

add_executable(testing testing.cpp matmul_quant.cpp matmul_kernels.cu)
target_compile_options(testing PUBLIC -fopenmp -mavx512f)
target_link_options(testing PUBLIC -lgomp)
target_link_libraries(testing "${TORCH_LIBRARIES}")
set_target_properties(testing PROPERTIES CXX_STANDARD 14)
set_target_properties(testing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(simple simple.cpp simple.cu)
target_link_libraries(simple "${TORCH_LIBRARIES}")
set_target_properties(simple PROPERTIES CXX_STANDARD 14)
set_target_properties(simple PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
