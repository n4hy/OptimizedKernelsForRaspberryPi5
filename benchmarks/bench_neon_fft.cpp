#include "bench_common.hpp"
#include <optmath/neon_kernels.hpp>
#include <optmath/radar_kernels.hpp>

using namespace optmath::neon;
using namespace optmath::radar;

// Benchmark: FIR filter
static void BM_NEON_FIR(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("NEON not available");
        return;
    }

    size_t signal_len = state.range(0);
    size_t kernel_len = state.range(1);

    Eigen::VectorXf x = bench::random_vector_f32(signal_len);
    Eigen::VectorXf h = bench::random_vector_f32(kernel_len);
    Eigen::VectorXf y;

    for (auto _ : state) {
        y = neon_fir(x, h);
        benchmark::DoNotOptimize(y.data());
    }

    // FIR: (signal_len - kernel_len + 1) * kernel_len * 2 flops
    size_t out_len = signal_len - kernel_len + 1;
    bench::set_flops(state, 2.0 * out_len * kernel_len);
}
BENCHMARK(BM_NEON_FIR)
    ->Args({1024, 16})
    ->Args({4096, 32})
    ->Args({16384, 64})
    ->Args({65536, 128});

// Benchmark: Cross-correlation
static void BM_NEON_XCorr(benchmark::State& state) {
    size_t N = state.range(0);

    Eigen::VectorXf x = bench::random_vector_f32(N);
    Eigen::VectorXf y = bench::random_vector_f32(N);
    Eigen::VectorXf result;

    for (auto _ : state) {
        result = xcorr(x, y);
        benchmark::DoNotOptimize(result.data());
    }

    // Approximately N^2 operations for full correlation
    bench::set_flops(state, 2.0 * N * N);
}
BENCHMARK(BM_NEON_XCorr)->RangeMultiplier(2)->Range(128, 4096);

// Benchmark: Complex cross-correlation
static void BM_NEON_XCorr_Complex(benchmark::State& state) {
    size_t N = state.range(0);

    Eigen::VectorXcf x = bench::random_vector_cf32(N);
    Eigen::VectorXcf y = bench::random_vector_cf32(N);
    Eigen::VectorXcf result;

    for (auto _ : state) {
        result = xcorr(x, y);
        benchmark::DoNotOptimize(result.data());
    }

    // Complex: 8 ops per multiply-add
    bench::set_flops(state, 8.0 * N * N);
}
BENCHMARK(BM_NEON_XCorr_Complex)->RangeMultiplier(2)->Range(128, 2048);

// Benchmark: Complex multiplication
static void BM_NEON_ComplexMul(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("NEON not available");
        return;
    }

    size_t N = state.range(0);

    Eigen::VectorXcf a = bench::random_vector_cf32(N);
    Eigen::VectorXcf b = bench::random_vector_cf32(N);
    Eigen::VectorXcf result;

    for (auto _ : state) {
        result = neon_complex_mul(a, b);
        benchmark::DoNotOptimize(result.data());
    }

    // 6 flops per complex multiply (4 mul + 2 add/sub)
    bench::set_flops(state, 6.0 * N);
}
BENCHMARK(BM_NEON_ComplexMul)->RangeMultiplier(4)->Range(256, 1048576);

// Benchmark: Complex magnitude
static void BM_NEON_ComplexMagnitude(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("NEON not available");
        return;
    }

    size_t N = state.range(0);

    Eigen::VectorXcf a = bench::random_vector_cf32(N);
    Eigen::VectorXf result;

    for (auto _ : state) {
        result = neon_complex_magnitude(a);
        benchmark::DoNotOptimize(result.data());
    }

    // 3 flops per magnitude (2 mul + 1 add + sqrt)
    bench::set_flops(state, 4.0 * N);  // sqrt counts as ~1 op
}
BENCHMARK(BM_NEON_ComplexMagnitude)->RangeMultiplier(4)->Range(256, 1048576);

// Benchmark: Window generation
static void BM_Window_Generate(benchmark::State& state) {
    size_t N = state.range(0);
    WindowType type = static_cast<WindowType>(state.range(1));
    Eigen::VectorXf window;

    for (auto _ : state) {
        window = generate_window(N, type);
        benchmark::DoNotOptimize(window.data());
    }

    bench::set_items_processed(state, N);
}
BENCHMARK(BM_Window_Generate)
    ->Args({1024, static_cast<int>(WindowType::HAMMING)})
    ->Args({1024, static_cast<int>(WindowType::HANNING)})
    ->Args({1024, static_cast<int>(WindowType::BLACKMAN)})
    ->Args({1024, static_cast<int>(WindowType::KAISER)})
    ->Args({4096, static_cast<int>(WindowType::HAMMING)})
    ->Args({16384, static_cast<int>(WindowType::HAMMING)});

// Benchmark: Window application
static void BM_Window_Apply(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("NEON not available");
        return;
    }

    size_t N = state.range(0);

    Eigen::VectorXf data = bench::random_vector_f32(N);
    Eigen::VectorXf window = generate_window(N, WindowType::HAMMING);

    for (auto _ : state) {
        Eigen::VectorXf temp = data;
        apply_window(temp, window);
        benchmark::DoNotOptimize(temp.data());
    }

    bench::set_flops(state, static_cast<double>(N));  // N multiplications
}
BENCHMARK(BM_Window_Apply)->RangeMultiplier(4)->Range(256, 1048576);

BENCHMARK_MAIN();
