#pragma once

#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <complex>

namespace bench {

// Random number generator for reproducible benchmarks
inline std::mt19937& get_rng() {
    static std::mt19937 rng(42);
    return rng;
}

// Generate random float vector
inline Eigen::VectorXf random_vector_f32(size_t n) {
    return Eigen::VectorXf::Random(n);
}

// Generate random float matrix
inline Eigen::MatrixXf random_matrix_f32(size_t rows, size_t cols) {
    return Eigen::MatrixXf::Random(rows, cols);
}

// Generate random complex vector
inline Eigen::VectorXcf random_vector_cf32(size_t n) {
    return Eigen::VectorXcf::Random(n);
}

// Generate random complex matrix
inline Eigen::MatrixXcf random_matrix_cf32(size_t rows, size_t cols) {
    return Eigen::MatrixXcf::Random(rows, cols);
}

// Common benchmark sizes
inline std::vector<int64_t> vector_sizes() {
    return {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576};
}

inline std::vector<std::pair<int64_t, int64_t>> matrix_sizes() {
    return {
        {32, 32}, {64, 64}, {128, 128}, {256, 256},
        {512, 512}, {1024, 1024}
    };
}

// Benchmark counter helpers
inline void set_items_processed(benchmark::State& state, size_t items) {
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * items);
}

inline void set_bytes_processed(benchmark::State& state, size_t bytes) {
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * bytes);
}

// FLOPS calculation helpers
inline void set_flops(benchmark::State& state, double flops_per_iter) {
    state.counters["FLOPS"] = benchmark::Counter(
        flops_per_iter,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000
    );
}

// For matrix multiply: 2*M*N*K flops
inline double gemm_flops(size_t M, size_t N, size_t K) {
    return 2.0 * M * N * K;
}

// For FFT: 5 * N * log2(N) flops (approximate)
inline double fft_flops(size_t N) {
    return 5.0 * N * std::log2(static_cast<double>(N));
}

// Prevent compiler from optimizing away results
template<typename T>
inline void do_not_optimize(T&& value) {
    benchmark::DoNotOptimize(value);
}

} // namespace bench
