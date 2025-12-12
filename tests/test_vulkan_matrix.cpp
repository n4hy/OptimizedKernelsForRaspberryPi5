#include <gtest/gtest.h>
#include <optmath/vulkan_backend.hpp>
#include <Eigen/Dense>

// Helper to check approximate equality
static void expect_approx_equal(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b, float tol = 1e-4) {
    ASSERT_EQ(a.rows(), b.rows());
    ASSERT_EQ(a.cols(), b.cols());
    for (int i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a(i), b(i), tol) << "at index " << i;
    }
}

TEST(VulkanMatrixTest, MatrixOperations) {
    if (!optmath::vulkan::is_available()) {
        GTEST_SKIP() << "Vulkan not available, skipping test.";
    }

    int M = 64;
    int N = 64;
    int K = 64;

    Eigen::MatrixXf A = Eigen::MatrixXf::Random(M, N);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(M, N);

    // Add
    {
        Eigen::MatrixXf expected = A + B;
        Eigen::MatrixXf result = optmath::vulkan::vulkan_mat_add(A, B);
        expect_approx_equal(result, expected);
    }

    // Sub
    {
        Eigen::MatrixXf expected = A - B;
        Eigen::MatrixXf result = optmath::vulkan::vulkan_mat_sub(A, B);
        expect_approx_equal(result, expected);
    }

    // Scale
    {
        float scalar = 2.5f;
        Eigen::MatrixXf expected = A * scalar;
        Eigen::MatrixXf result = optmath::vulkan::vulkan_mat_scale(A, scalar);
        expect_approx_equal(result, expected);
    }

    // Transpose
    {
        Eigen::MatrixXf expected = A.transpose();
        Eigen::MatrixXf result = optmath::vulkan::vulkan_mat_transpose(A);
        expect_approx_equal(result, expected);
    }

    // Mul
    {
        Eigen::MatrixXf MatA = Eigen::MatrixXf::Random(M, K);
        Eigen::MatrixXf MatB = Eigen::MatrixXf::Random(K, N);

        Eigen::MatrixXf expected = MatA * MatB;
        Eigen::MatrixXf result = optmath::vulkan::vulkan_mat_mul(MatA, MatB);
        // Matrix mul accumulates more error
        expect_approx_equal(result, expected, 1e-2);
    }
}
