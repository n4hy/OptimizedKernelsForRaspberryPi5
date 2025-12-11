#pragma once

#include <vector>
#include <cstddef>
#include <Eigen/Dense>

namespace optmath {
namespace neon {

    /**
     * @brief Checks if NEON acceleration was compiled in.
     */
    bool is_available();

    // --- Core Intrinsics Wrappers ---

    float neon_dot_f32(const float* a, const float* b, std::size_t n);
    double neon_dot_f64(const double* a, const double* b, std::size_t n);

    void neon_add_f32(float* out, const float* a, const float* b, std::size_t n);
    void neon_mul_f32(float* out, const float* a, const float* b, std::size_t n);

    // C += A * B (4x4 block)
    // A is 4x4 (row major or col major? Assuming col major for Eigen default, but standard microkernels often prefer row major packing.
    // Let's assume standard Eigen ColMajor storage for now or standard C-arrays.)
    // For simplicity in this demo, we will assume pointer input.
    void neon_gemm_4x4_f32(float* C, const float* A, const float* B, std::size_t ldc);

    void neon_fir_f32(const float* x, std::size_t n_x, const float* h, std::size_t n_h, float* y);

    void neon_relu_f32(float* data, std::size_t n);
    void neon_sigmoid_f32(float* data, std::size_t n);
    void neon_tanh_f32(float* data, std::size_t n);

    // --- Eigen Wrappers ---

    float neon_dot(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    double neon_dot(const Eigen::VectorXd& a, const Eigen::VectorXd& b);

    Eigen::VectorXf neon_add(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    Eigen::VectorXf neon_mul(const Eigen::VectorXf& a, const Eigen::VectorXf& b);

    // Computes y = x * h
    Eigen::VectorXf neon_fir(const Eigen::VectorXf& x, const Eigen::VectorXf& h);

    // In-place activations
    void neon_relu(Eigen::VectorXf& x);
    void neon_sigmoid(Eigen::VectorXf& x);
    void neon_tanh(Eigen::VectorXf& x);

    // Simple Matrix Multiplication wrapper (A * B)
    Eigen::MatrixXf neon_gemm(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B);

}
}
