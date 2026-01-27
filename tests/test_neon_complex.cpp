#include <gtest/gtest.h>
#include <optmath/neon_kernels.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <complex>

static void expect_approx_equal(const Eigen::VectorXf& a, const Eigen::VectorXf& b, float tol = 1e-4) {
    ASSERT_EQ(a.size(), b.size());
    for (int i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], b[i], tol) << "at index " << i;
    }
}

static void expect_approx_equal_complex(const Eigen::VectorXcf& a, const Eigen::VectorXcf& b, float tol = 1e-4) {
    ASSERT_EQ(a.size(), b.size());
    for (int i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i].real(), b[i].real(), tol) << "real part at index " << i;
        EXPECT_NEAR(a[i].imag(), b[i].imag(), tol) << "imag part at index " << i;
    }
}

TEST(NeonComplexTest, ComplexMultiplication) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 1024;
    Eigen::VectorXcf a = Eigen::VectorXcf::Random(N);
    Eigen::VectorXcf b = Eigen::VectorXcf::Random(N);

    // Reference: element-wise complex multiplication
    Eigen::VectorXcf expected = a.array() * b.array();

    // Test NEON implementation
    Eigen::VectorXcf result = optmath::neon::neon_complex_mul(a, b);

    expect_approx_equal_complex(result, expected);
}

TEST(NeonComplexTest, ComplexConjugateMultiplication) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 1024;
    Eigen::VectorXcf a = Eigen::VectorXcf::Random(N);
    Eigen::VectorXcf b = Eigen::VectorXcf::Random(N);

    // Reference: a * conj(b)
    Eigen::VectorXcf expected = a.array() * b.conjugate().array();

    // Test NEON implementation
    Eigen::VectorXcf result = optmath::neon::neon_complex_conj_mul(a, b);

    expect_approx_equal_complex(result, expected);
}

TEST(NeonComplexTest, ComplexDotProduct) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 1024;
    Eigen::VectorXcf a = Eigen::VectorXcf::Random(N);
    Eigen::VectorXcf b = Eigen::VectorXcf::Random(N);

    // Reference: sum(a * conj(b))
    std::complex<float> expected = a.dot(b);

    // Test NEON implementation
    std::complex<float> result = optmath::neon::neon_complex_dot(a, b);

    EXPECT_NEAR(result.real(), expected.real(), 1e-2 * N);
    EXPECT_NEAR(result.imag(), expected.imag(), 1e-2 * N);
}

TEST(NeonComplexTest, ComplexMagnitude) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 1024;
    Eigen::VectorXcf a = Eigen::VectorXcf::Random(N);

    // Reference
    Eigen::VectorXf expected = a.array().abs();

    // Test NEON implementation
    Eigen::VectorXf result = optmath::neon::neon_complex_magnitude(a);

    expect_approx_equal(result, expected, 1e-4);
}

TEST(NeonComplexTest, ComplexPhase) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 1024;
    Eigen::VectorXcf a = Eigen::VectorXcf::Random(N);

    // Reference
    Eigen::VectorXf expected(N);
    for (int i = 0; i < N; ++i) {
        expected[i] = std::arg(a[i]);
    }

    // Test NEON implementation
    Eigen::VectorXf result = optmath::neon::neon_complex_phase(a);

    expect_approx_equal(result, expected, 1e-5);
}

TEST(NeonComplexTest, InterleavedMultiplication) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 256;

    // Create interleaved complex arrays
    std::vector<float> a(2 * N), b(2 * N), result(2 * N), expected(2 * N);

    for (int i = 0; i < N; ++i) {
        a[2*i] = static_cast<float>(rand()) / RAND_MAX;
        a[2*i + 1] = static_cast<float>(rand()) / RAND_MAX;
        b[2*i] = static_cast<float>(rand()) / RAND_MAX;
        b[2*i + 1] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Reference
    for (int i = 0; i < N; ++i) {
        float ar = a[2*i], ai = a[2*i + 1];
        float br = b[2*i], bi = b[2*i + 1];
        expected[2*i] = ar * br - ai * bi;
        expected[2*i + 1] = ar * bi + ai * br;
    }

    // Test
    optmath::neon::neon_complex_mul_interleaved_f32(result.data(), a.data(), b.data(), N);

    for (int i = 0; i < 2 * N; ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-5) << "at index " << i;
    }
}

TEST(NeonComplexTest, MagnitudeSquared) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 1024;
    std::vector<float> re(N), im(N), result(N), expected(N);

    for (int i = 0; i < N; ++i) {
        re[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        im[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        expected[i] = re[i] * re[i] + im[i] * im[i];
    }

    optmath::neon::neon_complex_magnitude_squared_f32(result.data(), re.data(), im.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-6) << "at index " << i;
    }
}
