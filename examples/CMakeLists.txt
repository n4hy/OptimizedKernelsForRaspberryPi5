cmake_minimum_required(VERSION 3.18)

project(OptMathExamples)

add_executable(demo demo.cpp)

target_link_libraries(demo PRIVATE OptMathKernels::OptMathKernels)

# If you want to enable specific compiler flags for the example
target_compile_features(demo PRIVATE cxx_std_20)
