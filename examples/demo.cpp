#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <Eigen/Dense>

#include "optmath/neon_kernels.hpp"
#include "optmath/vulkan_backend.hpp"

using namespace optmath;

// Helper to fill vectors
void fill_random(Eigen::VectorXf& v) {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < v.size(); ++i) v[i] = dist(gen);
}

// Helper for timing
template<typename Func>
double time_ms(Func f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char** argv) {
    size_t N = 1000000; // 1M elements
    if (argc > 1) N = std::stoul(argv[1]);

    std::cout << "OptMathKernels Benchmark (N=" << N << ")\n";
    std::cout << "------------------------------------------\n";

    // --- NEON Availability ---
    if (neon::is_available()) {
        std::cout << "NEON: Available\n";
    } else {
        std::cout << "NEON: Not compiled or disabled\n";
    }

    // --- Vulkan Availability ---
    if (vulkan::is_available()) {
        std::cout << "Vulkan: Available (GPU initialized)\n";
    } else {
        std::cout << "Vulkan: Not available\n";
    }

    // Prepare Data
    Eigen::VectorXf a(N);
    Eigen::VectorXf b(N);
    fill_random(a);
    fill_random(b);

    // =========================================================================
    // 1. Dot Product
    // =========================================================================
    std::cout << "\n--- Dot Product ---\n";

    // CPU (Eigen)
    float dot_cpu = 0.0f;
    double t_cpu = time_ms([&]() { dot_cpu = a.dot(b); });
    std::cout << "Eigen (CPU): " << t_cpu << " ms, Result: " << dot_cpu << "\n";

    // NEON
    float dot_neon = 0.0f;
    double t_neon = time_ms([&]() { dot_neon = neon::neon_dot(a, b); });
    std::cout << "NEON       : " << t_neon << " ms, Result: " << dot_neon
              << " (Diff: " << std::abs(dot_cpu - dot_neon) << ")\n";

    // Vulkan
    if (vulkan::is_available()) {
        float dot_vk = 0.0f;
        double t_vk = time_ms([&]() { dot_vk = vulkan::vulkan_vec_dot(a, b); });
        std::cout << "Vulkan     : " << t_vk << " ms, Result: " << dot_vk
                  << " (Diff: " << std::abs(dot_cpu - dot_vk) << ")\n";
    }

    // =========================================================================
    // 2. Vector Addition
    // =========================================================================
    std::cout << "\n--- Vector Addition ---\n";

    // CPU
    Eigen::VectorXf add_cpu;
    t_cpu = time_ms([&]() { add_cpu = a + b; });
    std::cout << "Eigen (CPU): " << t_cpu << " ms\n";

    // NEON
    Eigen::VectorXf add_neon;
    t_neon = time_ms([&]() { add_neon = neon::neon_add(a, b); });
    float diff_neon = 0.0f;
    if (add_neon.size() == N) diff_neon = (add_cpu - add_neon).norm();
    std::cout << "NEON       : " << t_neon << " ms, Norm Diff: " << diff_neon << "\n";

    // Vulkan
    if (vulkan::is_available()) {
        Eigen::VectorXf add_vk;
        double t_vk = time_ms([&]() { add_vk = vulkan::vulkan_vec_add(a, b); });
        float diff_vk = 0.0f;
        if (add_vk.size() == N) diff_vk = (add_cpu - add_vk).norm();
        std::cout << "Vulkan     : " << t_vk << " ms, Norm Diff: " << diff_vk << "\n";
    }

    // =========================================================================
    // 3. FIR Filter
    // =========================================================================
    std::cout << "\n--- FIR Filter (Small Input) ---\n";
    size_t Nx = 10000;
    size_t Nh = 128;
    Eigen::VectorXf x(Nx), h(Nh);
    fill_random(x);
    fill_random(h);

    // CPU (Naive)
    Eigen::VectorXf fir_cpu;
    t_cpu = time_ms([&]() {
        // Naive CPU convolution
        size_t Ny = Nx - Nh + 1;
        fir_cpu.resize(Ny);
        for(size_t i=0; i<Ny; ++i) {
            float sum = 0.0f;
            for(size_t k=0; k<Nh; ++k) sum += x[i+k] * h[k];
            fir_cpu[i] = sum;
        }
    });
    std::cout << "Naive CPU  : " << t_cpu << " ms\n";

    // NEON
    Eigen::VectorXf fir_neon;
    t_neon = time_ms([&]() { fir_neon = neon::neon_fir(x, h); });
    diff_neon = 0.0f;
    if(fir_neon.size() == fir_cpu.size()) diff_neon = (fir_cpu - fir_neon).norm();
    std::cout << "NEON       : " << t_neon << " ms, Norm Diff: " << diff_neon << "\n";

     // Vulkan
    if (vulkan::is_available()) {
        Eigen::VectorXf fir_vk;
        double t_vk = time_ms([&]() { fir_vk = vulkan::vulkan_conv1d(x, h); });
        float diff_vk = 0.0f;
        if (fir_vk.size() == fir_cpu.size()) diff_vk = (fir_cpu - fir_vk).norm();
        std::cout << "Vulkan     : " << t_vk << " ms, Norm Diff: " << diff_vk << "\n";
    }

    return 0;
}
