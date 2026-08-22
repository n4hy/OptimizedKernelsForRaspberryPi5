#include <gtest/gtest.h>
#include <optmath/neon_kernels.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

TEST(NeonTranscendentalsTest, ExpApproximation) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 1024;
    std::vector<float> input(N), result(N);

    // Test range from -10 to 10
    for (int i = 0; i < N; ++i) {
        input[i] = -10.0f + 20.0f * i / (N - 1);
    }

    optmath::neon::neon_fast_exp_f32(result.data(), input.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = std::exp(input[i]);
        float rel_error = std::abs(result[i] - expected) / (std::abs(expected) + 1e-10f);
        // Base-2 range reduction + 6th-order minimax polynomial: ~1e-6 rel error.
        EXPECT_LT(rel_error, 1e-4f) << "at x = " << input[i]
                                    << ", expected = " << expected
                                    << ", got = " << result[i];
    }
}

TEST(NeonTranscendentalsTest, ExpBoundary) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    // Test boundary conditions
    std::vector<float> input = {0.0f, 1.0f, -1.0f, 88.0f, -88.0f, 100.0f, -100.0f};
    std::vector<float> result(input.size());

    optmath::neon::neon_fast_exp_f32(result.data(), input.data(), input.size());

    EXPECT_NEAR(result[0], 1.0f, 1e-4f);  // exp(0) = 1
    EXPECT_NEAR(result[1], std::exp(1.0f), 1e-3f);  // exp(1) ~= 2.72
    EXPECT_NEAR(result[2], std::exp(-1.0f), 1e-4f); // exp(-1) ~= 0.37
    EXPECT_GT(result[3], 0.0f);  // exp(88) should be large but finite
    EXPECT_GE(result[4], 0.0f);  // exp(-88) may underflow to 0 in fast approximation
    EXPECT_GT(result[5], 0.0f);  // exp(100) clamped to exp(88)
    EXPECT_GE(result[6], 0.0f);  // exp(-100) may underflow to 0 in fast approximation
}

TEST(NeonTranscendentalsTest, SinApproximation) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 1024;
    std::vector<float> input(N), result(N);

    // Test from -4*pi to 4*pi
    const float pi = 3.14159265358979323846f;
    for (int i = 0; i < N; ++i) {
        input[i] = -4.0f * pi + 8.0f * pi * i / (N - 1);
    }

    optmath::neon::neon_fast_sin_f32(result.data(), input.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = std::sin(input[i]);
        EXPECT_NEAR(result[i], expected, 1e-5f) << "at x = " << input[i];
    }
}

TEST(NeonTranscendentalsTest, CosApproximation) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 1024;
    std::vector<float> input(N), result(N);

    const float pi = 3.14159265358979323846f;
    for (int i = 0; i < N; ++i) {
        input[i] = -4.0f * pi + 8.0f * pi * i / (N - 1);
    }

    optmath::neon::neon_fast_cos_f32(result.data(), input.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = std::cos(input[i]);
        EXPECT_NEAR(result[i], expected, 1e-5f) << "at x = " << input[i];
    }
}

TEST(NeonTranscendentalsTest, SinCosLargeArgument) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    // Regression: the ONLY sin/cos coverage used to stop at 4*pi, so a broken
    // range reduction was invisible. Two defects lived in that blind spot:
    //   * PI_C was wrong in its 5th digit, leaving a 8.5e-12 residual in the
    //     four-part pi split that grew as 8.5e-12*k;
    //   * cos was computed as sin(x + pi/2), and forming that sum in float
    //     costs one ulp of x -- 4.8e-4 of error by |x| = 1e4.
    // The four-part split is exact for |k| < 2^15, i.e. |x| < ~1.0e5.
    const float limits[] = {1.0e2f, 1.0e3f, 1.0e4f, 1.0e5f};
    for (float R : limits) {
        const int N = 4096;
        std::vector<float> input(N), sin_out(N), cos_out(N);
        for (int i = 0; i < N; ++i) {
            input[i] = -R + 2.0f * R * i / (N - 1);
        }
        optmath::neon::neon_fast_sin_f32(sin_out.data(), input.data(), N);
        optmath::neon::neon_fast_cos_f32(cos_out.data(), input.data(), N);

        for (int i = 0; i < N; ++i) {
            const double x = static_cast<double>(input[i]);
            EXPECT_NEAR(sin_out[i], static_cast<float>(std::sin(x)), 1e-5f)
                << "sin at x = " << input[i] << " (range " << R << ")";
            EXPECT_NEAR(cos_out[i], static_cast<float>(std::cos(x)), 1e-5f)
                << "cos at x = " << input[i] << " (range " << R << ")";
        }
    }
}

TEST(NeonTranscendentalsTest, SinCosVectorAndScalarTailAgree) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    // The vectorized body (i + 3 < n) and the scalar tail are separate
    // implementations of the same reduction; a fix applied to one and not the
    // other would make results depend on the array length. Feeding n = 4 and
    // n = 1 exercises exactly one path each.
    const float xs[] = {0.3f, -2.7f, 12.5f, 137.0f, -981.25f, 12345.75f, -99999.5f};
    for (float x : xs) {
        float in4[4] = {x, x, x, x}, s4[4], c4[4];
        float in1[1] = {x}, s1[1], c1[1];
        optmath::neon::neon_fast_sin_f32(s4, in4, 4);
        optmath::neon::neon_fast_cos_f32(c4, in4, 4);
        optmath::neon::neon_fast_sin_f32(s1, in1, 1);
        optmath::neon::neon_fast_cos_f32(c1, in1, 1);
        // FMA in the vector path vs non-FMA in the tail permits ~1 ulp of drift.
        EXPECT_NEAR(s4[0], s1[0], 1e-6f) << "sin path mismatch at x = " << x;
        EXPECT_NEAR(c4[0], c1[0], 1e-6f) << "cos path mismatch at x = " << x;
    }
}

TEST(NeonTranscendentalsTest, SinCosIdentity) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 256;
    std::vector<float> input(N), sin_out(N), cos_out(N);

    const float pi = 3.14159265358979323846f;
    for (int i = 0; i < N; ++i) {
        input[i] = -2.0f * pi + 4.0f * pi * i / (N - 1);
    }

    optmath::neon::neon_fast_sin_f32(sin_out.data(), input.data(), N);
    optmath::neon::neon_fast_cos_f32(cos_out.data(), input.data(), N);

    // sin^2 + cos^2 = 1
    for (int i = 0; i < N; ++i) {
        float sum_sq = sin_out[i] * sin_out[i] + cos_out[i] * cos_out[i];
        EXPECT_NEAR(sum_sq, 1.0f, 1e-4f) << "at x = " << input[i];
    }
}

TEST(NeonTranscendentalsTest, SigmoidFast) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 1024;
    std::vector<float> input(N), result(N);

    // Test from -10 to 10
    for (int i = 0; i < N; ++i) {
        input[i] = -10.0f + 20.0f * i / (N - 1);
    }

    optmath::neon::neon_fast_sigmoid_f32(result.data(), input.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = 1.0f / (1.0f + std::exp(-input[i]));
        // Chains the (now accurate) fast exp; ~1e-6 error.
        EXPECT_NEAR(result[i], expected, 1e-4f) << "at x = " << input[i];
    }
}

TEST(NeonTranscendentalsTest, SigmoidProperties) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    std::vector<float> input = {0.0f, 10.0f, -10.0f, 1.0f, -1.0f};
    std::vector<float> result(input.size());

    optmath::neon::neon_fast_sigmoid_f32(result.data(), input.data(), input.size());

    EXPECT_NEAR(result[0], 0.5f, 1e-5f);  // sigmoid(0) = 0.5
    EXPECT_GT(result[1], 0.99f);           // sigmoid(10) close to 1
    EXPECT_LT(result[2], 0.01f);           // sigmoid(-10) close to 0

    // sigmoid(x) + sigmoid(-x) = 1
    EXPECT_NEAR(result[3] + result[4], 1.0f, 1e-4f);
}

TEST(NeonTranscendentalsTest, TanhFast) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    int N = 1024;
    std::vector<float> input(N), result(N);

    for (int i = 0; i < N; ++i) {
        input[i] = -5.0f + 10.0f * i / (N - 1);
    }

    optmath::neon::neon_fast_tanh_f32(result.data(), input.data(), N);

    for (int i = 0; i < N; ++i) {
        float expected = std::tanh(input[i]);
        // 2*sigmoid(2x)-1 with the now-accurate fast exp; ~1e-5 error.
        EXPECT_NEAR(result[i], expected, 1e-4f) << "at x = " << input[i];
    }
}

TEST(NeonTranscendentalsTest, TanhProperties) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    std::vector<float> input = {0.0f, 5.0f, -5.0f, 1.0f, -1.0f};
    std::vector<float> result(input.size());

    optmath::neon::neon_fast_tanh_f32(result.data(), input.data(), input.size());

    EXPECT_NEAR(result[0], 0.0f, 1e-5f);   // tanh(0) = 0
    EXPECT_GT(result[1], 0.99f);            // tanh(5) close to 1
    EXPECT_LT(result[2], -0.99f);           // tanh(-5) close to -1

    // tanh is odd: tanh(-x) = -tanh(x)
    EXPECT_NEAR(result[3], -result[4], 1e-5f);
}

TEST(NeonTranscendentalsTest, GEMMBlocked) {
    if (!optmath::neon::is_available()) {
        GTEST_SKIP() << "NEON not available, skipping test.";
    }

    // Test various matrix sizes
    std::vector<std::tuple<int, int, int>> sizes = {
        {16, 16, 16},
        {64, 64, 64},
        {128, 128, 128},
        {100, 50, 75},  // Non-power-of-2
        {17, 23, 31},   // Odd sizes
    };

    for (auto& [M, N, K] : sizes) {
        Eigen::MatrixXf A = Eigen::MatrixXf::Random(M, K);
        Eigen::MatrixXf B = Eigen::MatrixXf::Random(K, N);

        Eigen::MatrixXf expected = A * B;
        Eigen::MatrixXf result = optmath::neon::neon_gemm_blocked(A, B);

        ASSERT_EQ(result.rows(), expected.rows());
        ASSERT_EQ(result.cols(), expected.cols());

        float max_error = (result - expected).cwiseAbs().maxCoeff();
        float tol = 1e-3f * K;  // Error scales with K

        EXPECT_LT(max_error, tol) << "Matrix size: " << M << "x" << K << " * " << K << "x" << N;
    }
}
