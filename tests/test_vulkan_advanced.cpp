#include <gtest/gtest.h>
#include <optmath/vulkan_backend.hpp>
#include <Eigen/Dense>

TEST(VulkanReductionTest, ReductionsAndScan) {
    if (!optmath::vulkan::is_available()) {
        GTEST_SKIP() << "Vulkan not available, skipping test.";
    }

    int N = 256;
    Eigen::VectorXf a = Eigen::VectorXf::Random(N);
    // make positive for simpler checking
    a = a.cwiseAbs();

    // Sum
    {
        float expected = a.sum();
        float result = optmath::vulkan::vulkan_reduce_sum(a);
        EXPECT_NEAR(result, expected, 1e-1);
    }

    // Max
    {
        float expected = a.maxCoeff();
        float result = optmath::vulkan::vulkan_reduce_max(a);
        EXPECT_EQ(result, expected);
    }

    // Min
    {
        float expected = a.minCoeff();
        float result = optmath::vulkan::vulkan_reduce_min(a);
        EXPECT_EQ(result, expected);
    }

    // Scan (Prefix Sum)
    {
        // Exclusive scan expected
        Eigen::VectorXf expected(N);
        float sum = 0.0f;
        for (int i=0; i<N; ++i) {
            expected[i] = sum;
            sum += a[i];
        }

        Eigen::VectorXf result = optmath::vulkan::vulkan_scan_prefix_sum(a);
        ASSERT_EQ(result.size(), N);
        for(int i=0; i<N; ++i) {
             EXPECT_NEAR(result[i], expected[i], 1e-2) << "at index " << i;
        }
    }
}

TEST(VulkanFFTTest, BasicFFT) {
    if (!optmath::vulkan::is_available()) {
        GTEST_SKIP() << "Vulkan not available, skipping test.";
    }

    // Radix 2: Size 8
    {
        int N = 8;
        Eigen::VectorXf data(2*N); // Interleaved complex
        for(int i=0; i<N; ++i) {
            data[2*i] = (float)i; // Real
            data[2*i+1] = 0.0f;   // Imag
        }

        // Expected result for 0..7:
        // DC = 28.
        // Fwd
        Eigen::VectorXf gpu_data = data;
        optmath::vulkan::vulkan_fft_radix2(gpu_data, false);

        // Check DC (idx 0)
        EXPECT_NEAR(gpu_data[0], 28.0f, 1e-3);
        EXPECT_NEAR(gpu_data[1], 0.0f, 1e-3);

        // Inverse
        optmath::vulkan::vulkan_fft_radix2(gpu_data, true);

        // Scale by 1/N
        gpu_data /= (float)N;

        for(int i=0; i<N; ++i) {
            EXPECT_NEAR(gpu_data[2*i], data[2*i], 1e-3);
            EXPECT_NEAR(gpu_data[2*i+1], data[2*i+1], 1e-3);
        }
    }

    // Radix 4: Size 16
    {
        int N = 16;
        Eigen::VectorXf data(2*N);
        for(int i=0; i<N; ++i) {
            data[2*i] = 1.0f; // DC should be 16
            data[2*i+1] = 0.0f;
        }

        Eigen::VectorXf gpu_data = data;
        optmath::vulkan::vulkan_fft_radix4(gpu_data, false);

        EXPECT_NEAR(gpu_data[0], 16.0f, 1e-3);

        optmath::vulkan::vulkan_fft_radix4(gpu_data, true);
        gpu_data /= (float)N;

        for(int i=0; i<N; ++i) {
             EXPECT_NEAR(gpu_data[2*i], data[2*i], 1e-3);
        }
    }
}
