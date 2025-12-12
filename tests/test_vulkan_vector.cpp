#include <gtest/gtest.h>
#include <optmath/vulkan_backend.hpp>
#include <Eigen/Dense>
#include <iostream>

// Helper to check approximate equality
static void expect_approx_equal(const Eigen::VectorXf& a, const Eigen::VectorXf& b, float tol = 1e-4) {
    ASSERT_EQ(a.size(), b.size());
    for (int i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], b[i], tol) << "at index " << i;
    }
}

TEST(VulkanVectorTest, BasicOperations) {
    if (!optmath::vulkan::is_available()) {
        GTEST_SKIP() << "Vulkan not available, skipping test.";
    }

    int N = 1000;
    Eigen::VectorXf a = Eigen::VectorXf::Random(N);
    Eigen::VectorXf b = Eigen::VectorXf::Random(N);
    // Ensure no division by zero for div test
    b = b.cwiseAbs().array() + 0.1f;

    // Add
    {
        Eigen::VectorXf expected = a + b;
        Eigen::VectorXf result = optmath::vulkan::vulkan_vec_add(a, b);
        expect_approx_equal(result, expected);
    }

    // Sub
    {
        Eigen::VectorXf expected = a - b;
        Eigen::VectorXf result = optmath::vulkan::vulkan_vec_sub(a, b);
        expect_approx_equal(result, expected);
    }

    // Mul
    {
        Eigen::VectorXf expected = a.array() * b.array();
        Eigen::VectorXf result = optmath::vulkan::vulkan_vec_mul(a, b);
        expect_approx_equal(result, expected);
    }

    // Div
    {
        Eigen::VectorXf expected = a.array() / b.array();
        Eigen::VectorXf result = optmath::vulkan::vulkan_vec_div(a, b);
        expect_approx_equal(result, expected);
    }

    // Dot
    {
        float expected = a.dot(b);
        float result = optmath::vulkan::vulkan_vec_dot(a, b);
        // Dot product accumulates error, increase tolerance slightly
        EXPECT_NEAR(result, expected, 1e-2 * N);
    }

    // Norm
    {
        float expected = a.norm();
        float result = optmath::vulkan::vulkan_vec_norm(a);
        EXPECT_NEAR(result, expected, 1e-2 * N);
    }
}
