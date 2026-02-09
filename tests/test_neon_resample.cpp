#include <gtest/gtest.h>
#include <optmath/neon_kernels.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <numeric>

using namespace optmath::neon;

// Simple sinc-based lowpass filter for testing
static std::vector<float> design_sinc_filter(std::size_t len, float cutoff) {
    std::vector<float> h(len);
    int center = static_cast<int>(len) / 2;
    const float pi = static_cast<float>(M_PI);
    for (std::size_t i = 0; i < len; ++i) {
        float n = static_cast<float>(i) - center;
        if (std::fabs(n) < 1e-6f) {
            h[i] = cutoff;
        } else {
            h[i] = std::sin(pi * cutoff * n) / (pi * n);
        }
        // Hamming window
        h[i] *= 0.54f - 0.46f * std::cos(2.0f * pi * i / (len - 1));
    }
    return h;
}

TEST(NeonResampleTest, IdentityResample) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    // L=1, M=1 with a delta filter should pass through input
    const std::size_t N = 32;
    std::vector<float> in(N);
    for (std::size_t i = 0; i < N; ++i) in[i] = static_cast<float>(i + 1);

    // Delta filter: [1.0]
    float filter[] = {1.0f};

    std::vector<float> out(N + 10);
    std::size_t out_len = 0;

    neon_resample_oneshot_f32(out.data(), &out_len, in.data(), N, filter, 1, 1, 1);

    ASSERT_EQ(out_len, N);
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(out[i], in[i], 1e-5f) << "at index " << i;
    }
}

TEST(NeonResampleTest, UpsampleBy2OutputLength) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    const std::size_t N = 16;
    std::vector<float> in(N, 1.0f);

    // Simple 2-tap filter scaled by L=2
    float filter[] = {2.0f, 2.0f};
    std::size_t L = 2, M = 1;

    std::vector<float> out(N * L + 10);
    std::size_t out_len = 0;

    neon_resample_oneshot_f32(out.data(), &out_len, in.data(), N, filter, 2, L, M);

    EXPECT_EQ(out_len, N * L);
}

TEST(NeonResampleTest, DownsampleBy2OutputLength) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    const std::size_t N = 16;
    std::vector<float> in(N, 1.0f);

    float filter[] = {1.0f};
    std::size_t L = 1, M = 2;

    std::vector<float> out(N + 10);
    std::size_t out_len = 0;

    neon_resample_oneshot_f32(out.data(), &out_len, in.data(), N, filter, 1, L, M);

    EXPECT_EQ(out_len, N / 2);
}

TEST(NeonResampleTest, Rational3Over2OutputLength) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    const std::size_t N = 100;
    std::vector<float> in(N, 1.0f);
    std::size_t L = 3, M = 2;

    auto filter = design_sinc_filter(24, 1.0f / static_cast<float>(L));
    // Scale by L
    for (auto& v : filter) v *= static_cast<float>(L);

    std::vector<float> out(N * 2 + 20);
    std::size_t out_len = 0;

    neon_resample_oneshot_f32(out.data(), &out_len, in.data(), N,
                               filter.data(), filter.size(), L, M);

    // Expected output length: ceil(N * L / M) = ceil(150) = 150
    EXPECT_EQ(out_len, (N * L + M - 1) / M);
}

TEST(NeonResampleTest, DCGainPreservation) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    // A properly designed resampler should preserve DC level
    const std::size_t N = 200;
    const float dc_val = 3.5f;
    std::vector<float> in(N, dc_val);
    std::size_t L = 3, M = 2;

    // Design a proper lowpass filter
    auto filter = design_sinc_filter(48, 1.0f / static_cast<float>(L));
    // Scale by L for interpolation gain compensation
    for (auto& v : filter) v *= static_cast<float>(L);

    std::vector<float> out(N * 2 + 50);
    std::size_t out_len = 0;

    neon_resample_oneshot_f32(out.data(), &out_len, in.data(), N,
                               filter.data(), filter.size(), L, M);

    ASSERT_GT(out_len, 20u);

    // Check steady-state region (skip transients at edges)
    std::size_t skip = out_len / 4;
    for (std::size_t i = skip; i < out_len - skip; ++i) {
        EXPECT_NEAR(out[i], dc_val, 0.3f) << "at output index " << i;
    }
}

TEST(NeonResampleTest, StreamingConsistency) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    const std::size_t N = 64;
    std::vector<float> in(N);
    for (std::size_t i = 0; i < N; ++i) in[i] = std::sin(0.1f * i);

    float filter[] = {0.25f, 0.5f, 0.25f};
    std::size_t L = 2, M = 1;

    // One-shot result
    std::vector<float> out_oneshot(N * L + 10);
    std::size_t oneshot_len = 0;
    neon_resample_oneshot_f32(out_oneshot.data(), &oneshot_len, in.data(), N,
                               filter, 3, L, M);

    // Streaming result: process in two blocks
    PolyphaseResamplerState state;
    neon_resample_init(state, filter, 3, L, M);

    std::size_t block1 = 32;
    std::vector<float> out_stream(N * L + 10);
    std::size_t n1 = neon_resample_f32(out_stream.data(), in.data(), block1, state);
    std::size_t n2 = neon_resample_f32(out_stream.data() + n1, in.data() + block1, N - block1, state);

    EXPECT_EQ(n1 + n2, oneshot_len);

    for (std::size_t i = 0; i < oneshot_len; ++i) {
        EXPECT_NEAR(out_stream[i], out_oneshot[i], 1e-5f) << "at index " << i;
    }
}

TEST(NeonResampleTest, EigenWrapper) {
    if (!is_available()) {
        GTEST_SKIP() << "NEON not available";
    }

    const int N = 32;
    Eigen::VectorXf in = Eigen::VectorXf::Ones(N) * 2.0f;
    Eigen::VectorXf filter(1);
    filter << 1.0f;

    Eigen::VectorXf result = neon_resample(in, filter, 1, 1);

    ASSERT_EQ(result.size(), N);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(result[i], 2.0f, 1e-5f);
    }
}
