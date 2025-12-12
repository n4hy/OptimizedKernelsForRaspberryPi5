add_executable(test_basic test_basic.cpp)
target_link_libraries(test_basic PRIVATE OptMathKernels::OptMathKernels)
add_test(NAME test_basic COMMAND test_basic)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(test_vulkan_vector test_vulkan_vector.cpp)
target_link_libraries(test_vulkan_vector PRIVATE OptMathKernels::OptMathKernels GTest::gtest GTest::gtest_main)
add_test(NAME test_vulkan_vector COMMAND test_vulkan_vector)

add_executable(test_vulkan_matrix test_vulkan_matrix.cpp)
target_link_libraries(test_vulkan_matrix PRIVATE OptMathKernels::OptMathKernels GTest::gtest GTest::gtest_main)
add_test(NAME test_vulkan_matrix COMMAND test_vulkan_matrix)

add_executable(test_vulkan_dsp test_vulkan_dsp.cpp)
target_link_libraries(test_vulkan_dsp PRIVATE OptMathKernels::OptMathKernels GTest::gtest GTest::gtest_main)
add_test(NAME test_vulkan_dsp COMMAND test_vulkan_dsp)

add_executable(test_vulkan_advanced test_vulkan_advanced.cpp)
target_link_libraries(test_vulkan_advanced PRIVATE OptMathKernels::OptMathKernels GTest::gtest GTest::gtest_main)
add_test(NAME test_vulkan_advanced COMMAND test_vulkan_advanced)
