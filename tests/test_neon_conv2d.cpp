#include <gtest/gtest.h>
#include <optmath/neon_kernels.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

using namespace optmath::neon;

// Reference 2D convolution for verification
static void ref_conv2d(float* out, const float* in,
                       std::size_t in_rows, std::size_t in_cols,
                       const float* kernel, std::size_t kr, std::size_t kc) {
    std::size_t out_rows = in_rows - kr + 1;
    std::size_t out_cols = in_cols - kc + 1;
    for (std::size_t r = 0; r < out_rows; ++r) {
        for (std::size_t c = 0; c < out_cols; ++c) {
            float sum = 0.0f;
            for (std::size_t i = 0; i < kr; ++i) {
                for (std::size_t j = 0; j < kc; ++j) {
                    sum += in[(r + i) * in_cols + (c + j)] * kernel[i * kc + j];
                }
            }
            out[r * out_cols + c] = sum;
        }
    }
}

TEST(NeonConv2dTest, DeltaKernel) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    // A 1x1 kernel of value 1.0 should copy the input
    const std::size_t rows = 8, cols = 10;
    std::vector<float> in(rows * cols);
    for (std::size_t i = 0; i < rows * cols; ++i) in[i] = static_cast<float>(i);

    float kernel[] = {1.0f};
    std::vector<float> out(rows * cols);

    neon_conv2d_f32(out.data(), in.data(), rows, cols, kernel, 1, 1);

    for (std::size_t i = 0; i < rows * cols; ++i) {
        EXPECT_NEAR(out[i], in[i], 1e-5f) << "at index " << i;
    }
}

TEST(NeonConv2dTest, General3x3) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    const std::size_t rows = 16, cols = 20;
    std::vector<float> in(rows * cols);
    for (std::size_t i = 0; i < rows * cols; ++i) {
        in[i] = std::sin(static_cast<float>(i) * 0.1f);
    }

    // Sobel-like kernel
    float kernel[] = {
        -1.0f, 0.0f, 1.0f,
        -2.0f, 0.0f, 2.0f,
        -1.0f, 0.0f, 1.0f
    };

    std::size_t out_rows = rows - 2, out_cols = cols - 2;
    std::vector<float> out(out_rows * out_cols);
    std::vector<float> ref(out_rows * out_cols);

    neon_conv2d_f32(out.data(), in.data(), rows, cols, kernel, 3, 3);
    ref_conv2d(ref.data(), in.data(), rows, cols, kernel, 3, 3);

    for (std::size_t i = 0; i < out_rows * out_cols; ++i) {
        EXPECT_NEAR(out[i], ref[i], 1e-4f) << "at index " << i;
    }
}

TEST(NeonConv2dTest, Specialized3x3) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    const std::size_t rows = 32, cols = 32;
    std::vector<float> in(rows * cols);
    for (std::size_t i = 0; i < rows * cols; ++i) {
        in[i] = static_cast<float>(i % 17) - 8.0f;
    }

    // Gaussian-like 3x3 kernel
    float kernel[] = {
        1.0f / 16, 2.0f / 16, 1.0f / 16,
        2.0f / 16, 4.0f / 16, 2.0f / 16,
        1.0f / 16, 2.0f / 16, 1.0f / 16
    };

    std::size_t out_rows = rows - 2, out_cols = cols - 2;
    std::vector<float> out_general(out_rows * out_cols);
    std::vector<float> out_special(out_rows * out_cols);

    neon_conv2d_f32(out_general.data(), in.data(), rows, cols, kernel, 3, 3);
    neon_conv2d_3x3_f32(out_special.data(), in.data(), rows, cols, kernel);

    for (std::size_t i = 0; i < out_rows * out_cols; ++i) {
        EXPECT_NEAR(out_special[i], out_general[i], 1e-4f) << "at index " << i;
    }
}

TEST(NeonConv2dTest, Specialized5x5) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    const std::size_t rows = 32, cols = 32;
    std::vector<float> in(rows * cols);
    for (std::size_t i = 0; i < rows * cols; ++i) {
        in[i] = std::cos(static_cast<float>(i) * 0.05f);
    }

    // 5x5 box blur kernel
    float kernel[25];
    for (int i = 0; i < 25; ++i) kernel[i] = 1.0f / 25.0f;

    std::size_t out_rows = rows - 4, out_cols = cols - 4;
    std::vector<float> out_general(out_rows * out_cols);
    std::vector<float> out_special(out_rows * out_cols);

    neon_conv2d_f32(out_general.data(), in.data(), rows, cols, kernel, 5, 5);
    neon_conv2d_5x5_f32(out_special.data(), in.data(), rows, cols, kernel);

    for (std::size_t i = 0; i < out_rows * out_cols; ++i) {
        EXPECT_NEAR(out_special[i], out_general[i], 1e-4f) << "at index " << i;
    }
}

TEST(NeonConv2dTest, SeparableMatchesGeneral) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    const std::size_t rows = 24, cols = 24;
    std::vector<float> in(rows * cols);
    for (std::size_t i = 0; i < rows * cols; ++i) {
        in[i] = static_cast<float>(i % 13) * 0.5f - 3.0f;
    }

    // Separable kernel: row = [1, 2, 1]/4, col = [1, 2, 1]/4
    // Full 2D kernel = outer product
    float row_k[] = {0.25f, 0.5f, 0.25f};
    float col_k[] = {0.25f, 0.5f, 0.25f};

    float full_k[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            full_k[r * 3 + c] = col_k[r] * row_k[c];

    std::size_t out_rows = rows - 2, out_cols = cols - 2;
    std::vector<float> out_general(out_rows * out_cols);
    std::vector<float> out_separable(out_rows * out_cols);

    neon_conv2d_f32(out_general.data(), in.data(), rows, cols, full_k, 3, 3);
    neon_conv2d_separable_f32(out_separable.data(), in.data(), rows, cols,
                               row_k, 3, col_k, 3);

    for (std::size_t i = 0; i < out_rows * out_cols; ++i) {
        EXPECT_NEAR(out_separable[i], out_general[i], 1e-3f) << "at index " << i;
    }
}

TEST(NeonConv2dTest, BoxBlur) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    // A constant input through a box blur should produce the same constant
    const std::size_t rows = 16, cols = 16;
    const float val = 7.0f;
    std::vector<float> in(rows * cols, val);

    float kernel[9];
    for (int i = 0; i < 9; ++i) kernel[i] = 1.0f / 9.0f;

    std::size_t out_rows = rows - 2, out_cols = cols - 2;
    std::vector<float> out(out_rows * out_cols);

    neon_conv2d_3x3_f32(out.data(), in.data(), rows, cols, kernel);

    for (std::size_t i = 0; i < out_rows * out_cols; ++i) {
        EXPECT_NEAR(out[i], val, 1e-4f) << "at index " << i;
    }
}

TEST(NeonConv2dTest, NonSquareInput) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    // Test with non-square input dimensions
    const std::size_t rows = 10, cols = 50;
    std::vector<float> in(rows * cols);
    for (std::size_t i = 0; i < rows * cols; ++i) {
        in[i] = static_cast<float>(i) * 0.01f;
    }

    float kernel[] = {1.0f, -1.0f, 0.0f, -1.0f, 1.0f, 0.0f};  // 2x3 kernel

    std::size_t out_rows = rows - 1, out_cols = cols - 2;
    std::vector<float> out(out_rows * out_cols);
    std::vector<float> ref(out_rows * out_cols);

    neon_conv2d_f32(out.data(), in.data(), rows, cols, kernel, 2, 3);
    ref_conv2d(ref.data(), in.data(), rows, cols, kernel, 2, 3);

    for (std::size_t i = 0; i < out_rows * out_cols; ++i) {
        EXPECT_NEAR(out[i], ref[i], 1e-3f) << "at index " << i;
    }
}

TEST(NeonConv2dTest, SmallInput) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    // Edge case: input just barely fits the kernel
    const std::size_t rows = 3, cols = 3;
    float in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float kernel[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    float out[1];
    neon_conv2d_3x3_f32(out, in, rows, cols, kernel);

    // 1*1 + 0*2 + 0*3 + 0*4 + 1*5 + 0*6 + 0*7 + 0*8 + 1*9 = 15
    EXPECT_NEAR(out[0], 15.0f, 1e-5f);
}

TEST(NeonConv2dTest, EigenWrapper) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    Eigen::MatrixXf in = Eigen::MatrixXf::Ones(8, 8) * 3.0f;
    Eigen::MatrixXf kernel = Eigen::MatrixXf::Ones(3, 3) / 9.0f;

    Eigen::MatrixXf result = neon_conv2d(in, kernel);

    ASSERT_EQ(result.rows(), 6);
    ASSERT_EQ(result.cols(), 6);

    for (int r = 0; r < result.rows(); ++r) {
        for (int c = 0; c < result.cols(); ++c) {
            EXPECT_NEAR(result(r, c), 3.0f, 1e-4f);
        }
    }
}
