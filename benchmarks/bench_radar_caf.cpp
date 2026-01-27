#include "bench_common.hpp"
#include <optmath/radar_kernels.hpp>
#include <optmath/neon_kernels.hpp>

using namespace optmath::radar;
using namespace optmath::neon;

// Benchmark: Cross-Ambiguity Function
static void BM_Radar_CAF(benchmark::State& state) {
    size_t n_samples = state.range(0);
    size_t n_doppler = state.range(1);
    size_t n_range = state.range(2);

    float sample_rate = 1e6f;
    float doppler_start = -500.0f;
    float doppler_step = 50.0f;

    Eigen::VectorXcf ref = bench::random_vector_cf32(n_samples);
    Eigen::VectorXcf surv = bench::random_vector_cf32(n_samples);
    Eigen::MatrixXf result;

    for (auto _ : state) {
        result = caf(ref, surv, n_doppler, doppler_start, doppler_step, sample_rate, n_range);
        benchmark::DoNotOptimize(result.data());
    }

    // Approximate flops: n_doppler * n_range * n_samples * 10 (Doppler shift + correlation)
    bench::set_flops(state, 10.0 * n_doppler * n_range * n_samples);
}
BENCHMARK(BM_Radar_CAF)
    ->Args({1024, 21, 50})     // Small
    ->Args({4096, 41, 100})    // Medium
    ->Args({16384, 61, 200})   // Large
    ->Args({65536, 101, 500}); // Very large

// Benchmark: 1D CA-CFAR
static void BM_Radar_CFAR_CA(benchmark::State& state) {
    size_t N = state.range(0);
    size_t guard = state.range(1);
    size_t ref = state.range(2);

    Eigen::VectorXf data = bench::random_vector_f32(N).cwiseAbs();
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> detections;

    for (auto _ : state) {
        detections = cfar_ca(data, guard, ref, 10.0f);
        benchmark::DoNotOptimize(detections.data());
    }

    // N cells, 2*ref averaging per cell
    bench::set_flops(state, static_cast<double>(N * 2 * ref));
}
BENCHMARK(BM_Radar_CFAR_CA)
    ->Args({1024, 2, 8})
    ->Args({4096, 4, 16})
    ->Args({16384, 4, 32})
    ->Args({65536, 8, 64});

// Benchmark: 2D CA-CFAR
static void BM_Radar_CFAR_2D(benchmark::State& state) {
    size_t n_doppler = state.range(0);
    size_t n_range = state.range(1);

    Eigen::MatrixXf data = bench::random_matrix_f32(n_doppler, n_range).cwiseAbs();
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> detections;

    size_t guard_range = 2, guard_doppler = 2;
    size_t ref_range = 4, ref_doppler = 4;

    for (auto _ : state) {
        detections = cfar_2d(data, guard_range, guard_doppler, ref_range, ref_doppler, 10.0f);
        benchmark::DoNotOptimize(detections.data());
    }

    // Each cell averages ~4 * ref_range * ref_doppler cells
    bench::set_flops(state, static_cast<double>(n_doppler * n_range * 4 * ref_range * ref_doppler));
}
BENCHMARK(BM_Radar_CFAR_2D)
    ->Args({64, 128})
    ->Args({128, 256})
    ->Args({256, 512})
    ->Args({512, 1024});

// Benchmark: NLMS adaptive filter
static void BM_Radar_NLMS(benchmark::State& state) {
    size_t N = state.range(0);
    size_t filter_len = state.range(1);

    Eigen::VectorXf input = bench::random_vector_f32(N);
    Eigen::VectorXf reference = bench::random_vector_f32(N);
    Eigen::VectorXf output;

    for (auto _ : state) {
        output = nlms_filter(input, reference, filter_len, 0.1f, 1e-6f);
        benchmark::DoNotOptimize(output.data());
    }

    // Per sample: filter_len muls + filter_len updates
    bench::set_flops(state, 4.0 * N * filter_len);
}
BENCHMARK(BM_Radar_NLMS)
    ->Args({4096, 16})
    ->Args({16384, 32})
    ->Args({65536, 64})
    ->Args({262144, 128});

// Benchmark: MTI filter
static void BM_Radar_MTI(benchmark::State& state) {
    size_t n_pulses = state.range(0);
    size_t n_range = state.range(1);

    Eigen::MatrixXf data = bench::random_matrix_f32(n_pulses, n_range);
    Eigen::VectorXf coeffs(3);
    coeffs << 1.0f, -2.0f, 1.0f;  // 3-pulse canceller

    Eigen::MatrixXf output;

    for (auto _ : state) {
        output = mti_filter(data, coeffs);
        benchmark::DoNotOptimize(output.data());
    }

    // (n_pulses - n_coeffs + 1) * n_range * n_coeffs * 2 flops
    bench::set_flops(state, 2.0 * (n_pulses - 2) * n_range * 3);
}
BENCHMARK(BM_Radar_MTI)
    ->Args({32, 256})
    ->Args({64, 512})
    ->Args({128, 1024})
    ->Args({256, 2048});

// Benchmark: Beamforming (delay-and-sum)
static void BM_Radar_Beamform_DelaySum(benchmark::State& state) {
    size_t n_channels = state.range(0);
    size_t n_samples = state.range(1);

    Eigen::MatrixXf inputs = bench::random_matrix_f32(n_channels, n_samples);
    Eigen::VectorXi delays = Eigen::VectorXi::Zero(n_channels);
    for (size_t i = 0; i < n_channels; ++i) {
        delays[i] = static_cast<int>(i * 2);
    }
    Eigen::VectorXf weights = Eigen::VectorXf::Ones(n_channels);

    Eigen::VectorXf output;

    for (auto _ : state) {
        output = beamform_delay_sum(inputs, delays, weights);
        benchmark::DoNotOptimize(output.data());
    }

    bench::set_flops(state, 2.0 * n_channels * n_samples);
}
BENCHMARK(BM_Radar_Beamform_DelaySum)
    ->Args({4, 4096})
    ->Args({8, 16384})
    ->Args({16, 65536})
    ->Args({32, 262144});

// Benchmark: Beamforming (phase-shift)
static void BM_Radar_Beamform_Phase(benchmark::State& state) {
    if (!is_available()) {
        state.SkipWithError("NEON not available");
        return;
    }

    size_t n_channels = state.range(0);
    size_t n_samples = state.range(1);

    Eigen::MatrixXcf inputs = bench::random_matrix_cf32(n_channels, n_samples);
    Eigen::VectorXf phases = Eigen::VectorXf::LinSpaced(n_channels, 0, 3.14159f);
    Eigen::VectorXf weights = Eigen::VectorXf::Ones(n_channels);

    Eigen::VectorXcf output;

    for (auto _ : state) {
        output = beamform_phase(inputs, phases, weights);
        benchmark::DoNotOptimize(output.data());
    }

    // Complex operations: 8 flops per sample per channel
    bench::set_flops(state, 8.0 * n_channels * n_samples);
}
BENCHMARK(BM_Radar_Beamform_Phase)
    ->Args({4, 4096})
    ->Args({8, 16384})
    ->Args({16, 65536});

// Benchmark: Steering vector generation
static void BM_Radar_SteeringVector(benchmark::State& state) {
    size_t n_elements = state.range(0);
    float d_lambda = 0.5f;

    Eigen::VectorXcf steering;

    for (auto _ : state) {
        for (float theta = -1.5f; theta <= 1.5f; theta += 0.1f) {
            steering = steering_vector_ula(n_elements, d_lambda, theta);
            benchmark::DoNotOptimize(steering.data());
        }
    }

    bench::set_items_processed(state, 31 * n_elements);  // ~31 angles
}
BENCHMARK(BM_Radar_SteeringVector)->RangeMultiplier(2)->Range(4, 64);

BENCHMARK_MAIN();
