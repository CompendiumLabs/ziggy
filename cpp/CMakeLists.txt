cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test)

find_package(Torch REQUIRED)

add_executable(testing testing.cpp matmul_quant.cpp)
target_compile_options(testing PUBLIC -fopenmp -mavx512f)
target_link_options(testing PUBLIC -lgomp)
target_link_libraries(testing "${TORCH_LIBRARIES}")
set_target_properties(testing PROPERTIES CXX_STANDARD 14)
