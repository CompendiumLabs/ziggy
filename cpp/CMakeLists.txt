cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test)

find_package(Torch REQUIRED)

add_executable(matmul matmul.cpp)
target_compile_options(matmul PUBLIC -fopenmp)
target_link_options(matmul PUBLIC -lgomp)
target_link_libraries(matmul "${TORCH_LIBRARIES}")
set_target_properties(matmul PROPERTIES CXX_STANDARD 17)
