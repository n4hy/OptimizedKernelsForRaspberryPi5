# OptMathKernels

**High-Performance Numerical Library for Raspberry Pi 5 and NVIDIA GPUs**

OptMathKernels is a C++20 numerical library optimized for **Raspberry Pi 5** and **NVIDIA CUDA GPUs**. It seamlessly bridges **Eigen** (CPU), **ARM NEON** (SIMD), **Vulkan** (Compute Shaders), and **CUDA** (NVIDIA GPUs) into a single, easy-to-use API.

Designed to accelerate math and signal processing tasks by leveraging:
- **Raspberry Pi 5**: Cortex-A76 NEON and VideoCore VII GPU
- **NVIDIA GPUs**: cuBLAS, cuFFT, cuSOLVER, and Tensor Cores (Volta+)

While remaining compatible with standard Linux x86/ARM environments.

---

## Key Applications

### Passive Radar Signal Processing
OptMathKernels powers the [PassiveRadar_Kraken](https://github.com/n4hy/PassiveRadar_Kraken) project, providing hardware-accelerated kernels for:

**ARM NEON (Raspberry Pi 5):**
| Operation | Speedup | Application |
|-----------|---------|-------------|
| Complex multiply | 4-8x | CAF Doppler shifting |
| Dot product | 4-6x | NLMS adaptive filter |
| GEMM (blocked) | 3-5x | Beamforming, covariance |
| Transcendentals | 10-50x | Phase computation |
| FFT (Vulkan) | 10x | Large batch processing |

**NVIDIA CUDA (Workstation/Server):**
| Operation | Speedup | GPU Features |
|-----------|---------|--------------|
| GEMM (1024x1024) | 50-100x | Tensor Cores (Ampere+) |
| FFT (64K points) | 20-50x | cuFFT optimized |
| CAF (full map) | 30-80x | Batched FFT + complex ops |
| CFAR 2D | 10-30x | Parallel threshold compute |
| Beamforming | 20-50x | cuBLAS matrix-vector |
| Transcendentals | 50-200x | CUDA fast math intrinsics |

### General Numerical Computing
- Machine learning inference (activation functions, matrix ops)
- Digital signal processing (filtering, FFT, convolution)
- Scientific computing (linear algebra, statistics)
- Real-time audio/video processing

---

## Features

### Core Capabilities
- **NEON Acceleration**: Hand-tuned ARMv8 NEON intrinsics for SIMD acceleration on 64-bit ARM (aarch64). Includes optimized matrix multiplication, convolution, and vector math.
- **Vulkan Compute**: Massive parallel offloading to the GPU (VideoCore VII on Pi 5). Supports large vector operations, matrix math, FFT (Radix-2/4), and reductions.
- **Eigen Integration**: Fully compatible with `Eigen::VectorXf`, `Eigen::MatrixXf`, and `Eigen::VectorXcf`. Pass your existing data structures directly to accelerated kernels.
- **Easy Integration**: Standard CMake package that installs to `/usr/local` and works with `find_package(OptMathKernels)`.

### Radar Signal Processing (`optmath::radar`)
- **Cross-Ambiguity Function (CAF)**: Core passive radar operation for range-Doppler detection
- **CFAR Detection**: 1D/2D Cell-Averaging and Ordered Statistic CFAR detectors
- **Clutter Filtering**: NLMS adaptive filter and projection-based clutter cancellation
- **Doppler Processing**: FFT-based Doppler processing and MTI filters
- **Beamforming**: Delay-and-sum and phase-shift beamformers with steering vector generation
- **Window Functions**: Hamming, Hanning, Blackman, Blackman-Harris, Kaiser

### Optimized Kernels (`optmath::neon`)
- **Vectorized Transcendentals**: Fast NEON-accelerated exp, sin, cos, sigmoid, tanh (~10-50x faster than scalar)
- **Complex Operations**: Vectorized complex multiply, conjugate multiply, magnitude, phase
- **Cache-Blocked GEMM**: 8x8 microkernel with MC=128, KC=256 blocking for Cortex-A76
- **Reductions**: Sum, max, min, dot product with horizontal NEON adds

### GPU Acceleration (`optmath::vulkan`)
- **Tiled GPU Matrix Multiply**: 16x16 shared memory tiles for efficient GPU GEMM
- **FFT**: Radix-2/4 FFT with butterfly operations in compute shaders
- **Convolution**: 1D and 2D convolution with separable kernel optimization
- **Vector Operations**: Add, multiply, dot product, reductions

### NVIDIA CUDA Acceleration (`optmath::cuda`) - **NEW!**
- **cuBLAS Integration**: Level 1, 2, and 3 BLAS operations with Tensor Core support
- **cuFFT**: High-performance FFT up to 64M points, batched transforms
- **cuSOLVER**: Cholesky, LU, QR, SVD, eigenvalue decomposition
- **Tensor Cores**: Automatic acceleration on Volta+ (SM 7.0+) and Ampere+ (SM 8.0+)
- **Mixed Precision**: FP32, TF32, FP16 compute modes
- **Multi-GPU**: Device enumeration, peer-to-peer access, workload distribution
- **Unified Memory**: Simplified CPU-GPU data management
- **Radar Processing**: Full GPU-accelerated CAF, CFAR, beamforming, NLMS filter

---

## Prerequisites

### Raspberry Pi 5 / Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libeigen3-dev \
    libvulkan-dev \
    mesa-vulkan-drivers \
    vulkan-tools \
    glslang-tools \
    glslc
```

### NVIDIA CUDA Support (Optional)

For NVIDIA GPU acceleration, install the CUDA Toolkit:

```bash
# Ubuntu/Debian with NVIDIA driver already installed
sudo apt install -y nvidia-cuda-toolkit

# Or download from NVIDIA (recommended for latest version)
# https://developer.nvidia.com/cuda-downloads

# Verify installation
nvcc --version
nvidia-smi
```

Supported RTX architectures (default build):
| Architecture | SM | GPUs | Features |
|--------------|-----|------|----------|
| **Turing** | 7.5 | RTX 2080 Ti, 2080, 2070, 2060 | Tensor Cores Gen 1 |
| **Ampere** | 8.0/8.6 | RTX 3090, 3080, 3070, 3060 | Tensor Cores Gen 3, TF32 |
| **Ada Lovelace** | 8.9 | RTX 4090, 4080, 4070, 4060 | Tensor Cores Gen 4, FP8 |
| **Blackwell** | 10.0 | **RTX 5090**, 5080, 5070 | Tensor Cores Gen 5, FP8, 32GB |

Data center GPUs (add manually):
- Volta (SM 7.0): V100, Titan V
- Hopper (SM 9.0): H100

> **Note:** GoogleTest and Google Benchmark are automatically fetched during the build (via CMake `FetchContent`).

---

## Build & Install

### 1. Clone the Repository
```bash
git clone https://github.com/n4hy/OptimizedKernelsForRaspberryPi5.git
cd OptimizedKernelsForRaspberryPi5
```

### 2. Configure and Build

**Raspberry Pi 5 (NEON + Vulkan):**
```bash
mkdir -p build && cd build

cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_NEON=ON \
      -DENABLE_VULKAN=ON \
      -DENABLE_CUDA=OFF \
      -DBUILD_TESTS=ON \
      -DBUILD_BENCHMARKS=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      ..

make -j$(nproc)
```

**NVIDIA Workstation (CUDA + optional Vulkan):**
```bash
mkdir -p build && cd build

cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_NEON=OFF \
      -DENABLE_VULKAN=ON \
      -DENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
      -DBUILD_TESTS=ON \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      ..

make -j$(nproc)
```

**All backends (for development/testing):**
```bash
cmake -DENABLE_NEON=ON -DENABLE_VULKAN=ON -DENABLE_CUDA=ON ..
```

### 3. Run Tests
```bash
ctest --output-on-failure
```

### 4. Install
```bash
sudo make install
```

### 5. Build Benchmarks (Optional)
```bash
cmake -DBUILD_BENCHMARKS=ON ..
make -j$(nproc)
./benchmarks/bench_neon_transcendentals
./benchmarks/bench_radar_caf
```

---

## Usage Guide

### CMake Integration

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyApp)

find_package(OptMathKernels REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE
    OptMathKernels::OptMathKernels
    Eigen3::Eigen
)
```

### PassiveRadar_Kraken Integration

OptMathKernels integrates automatically with PassiveRadar_Kraken when installed:

```bash
# Build PassiveRadar_Kraken with OptMathKernels
cd /path/to/PassiveRadar_Kraken/src
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
# CMake output: "OptMathKernels found - enabling NEON optimization"
make -j$(nproc)
```

The following components are accelerated:
- `caf_processing.cpp` - NEON complex multiply for Doppler shifts
- `eca_b_clutter_canceller.cpp` - NEON dot products for NLMS filter

---

## API Reference

### NEON Kernels (`optmath::neon`)

#### Vector Operations
```cpp
#include <optmath/neon_kernels.hpp>

if (optmath::neon::is_available()) {
    Eigen::VectorXf a = Eigen::VectorXf::Random(1024);
    Eigen::VectorXf b = Eigen::VectorXf::Random(1024);

    // Basic operations
    Eigen::VectorXf c = optmath::neon::neon_add(a, b);
    Eigen::VectorXf d = optmath::neon::neon_mul(a, b);
    float dot = optmath::neon::neon_dot(a, b);
    float norm = optmath::neon::neon_norm(a);

    // Reductions
    float sum = optmath::neon::neon_reduce_sum(a);
    float max = optmath::neon::neon_reduce_max(a);
}
```

#### Matrix Operations
```cpp
Eigen::MatrixXf A = Eigen::MatrixXf::Random(256, 256);
Eigen::MatrixXf B = Eigen::MatrixXf::Random(256, 256);

// Basic GEMM (4x4 tiled)
Eigen::MatrixXf C = optmath::neon::neon_gemm(A, B);

// Optimized blocked GEMM (8x8 microkernel, cache blocking)
Eigen::MatrixXf D = optmath::neon::neon_gemm_blocked(A, B);

// Other operations
Eigen::MatrixXf At = optmath::neon::neon_mat_transpose(A);
Eigen::MatrixXf As = optmath::neon::neon_mat_scale(A, 2.5f);
Eigen::VectorXf Av = optmath::neon::neon_mat_vec_mul(A, v);
```

#### Vectorized Transcendentals
```cpp
std::vector<float> input(N), output(N);

// Fast exp approximation (~1e-6 relative error)
optmath::neon::neon_exp_f32_approx(output.data(), input.data(), N);

// Fast sin/cos approximation
optmath::neon::neon_sin_f32_approx(output.data(), input.data(), N);
optmath::neon::neon_cos_f32_approx(output.data(), input.data(), N);

// Fast activation functions
optmath::neon::neon_sigmoid_f32_fast(output.data(), input.data(), N);
optmath::neon::neon_tanh_f32_fast(output.data(), input.data(), N);
optmath::neon::neon_relu_f32(data.data(), N);  // In-place
```

#### Complex Number Operations
```cpp
Eigen::VectorXcf a = Eigen::VectorXcf::Random(1024);
Eigen::VectorXcf b = Eigen::VectorXcf::Random(1024);

// Element-wise complex operations
Eigen::VectorXcf c = optmath::neon::neon_complex_mul(a, b);
Eigen::VectorXcf d = optmath::neon::neon_complex_conj_mul(a, b);  // a * conj(b)

// Complex dot product: sum(a * conj(b))
std::complex<float> dot = optmath::neon::neon_complex_dot(a, b);

// Magnitude and phase
Eigen::VectorXf mag = optmath::neon::neon_complex_magnitude(a);
Eigen::VectorXf phase = optmath::neon::neon_complex_phase(a);
```

#### Deinterleaved Complex Operations (for C libraries)
```cpp
// For ctypes/C interop where complex data is in separate real/imag arrays
float* out_re, *out_im;
const float* a_re, *a_im, *b_re, *b_im;

optmath::neon::neon_complex_mul_f32(
    out_re, out_im,     // Output
    a_re, a_im,         // Input A
    b_re, b_im,         // Input B
    n_samples
);

float dot_re, dot_im;
optmath::neon::neon_complex_dot_f32(
    &dot_re, &dot_im,
    a_re, a_im,
    b_re, b_im,
    n_samples
);
```

---

### Radar Kernels (`optmath::radar`)

#### Window Functions
```cpp
#include <optmath/radar_kernels.hpp>
using namespace optmath::radar;

// Generate window
Eigen::VectorXf window = generate_window(1024, WindowType::HAMMING);
// Available: RECTANGULAR, HAMMING, HANNING, BLACKMAN, BLACKMAN_HARRIS, KAISER

// Apply window to data
Eigen::VectorXf data = Eigen::VectorXf::Random(1024);
apply_window(data, window);

// Apply to complex data
Eigen::VectorXcf complex_data = Eigen::VectorXcf::Random(1024);
apply_window(complex_data, window);
```

#### Cross-Correlation
```cpp
Eigen::VectorXf x = /* signal */;
Eigen::VectorXf y = /* reference */;

// Real cross-correlation
Eigen::VectorXf corr = xcorr(x, y);  // Size: x.size() + y.size() - 1

// Complex cross-correlation
Eigen::VectorXcf cx = /* complex signal */;
Eigen::VectorXcf cy = /* complex reference */;
Eigen::VectorXcf ccorr = xcorr(cx, cy);
```

#### Cross-Ambiguity Function (CAF)
```cpp
// Reference and surveillance signals
Eigen::VectorXcf ref = /* transmitter signal */;
Eigen::VectorXcf surv = /* receiver signal */;

// CAF parameters
size_t n_doppler_bins = 101;
size_t n_range_bins = 500;
float doppler_start = -500.0f;  // Hz
float doppler_step = 10.0f;      // Hz
float sample_rate = 1e6f;        // Hz

// Compute CAF (output: n_doppler x n_range magnitude matrix)
Eigen::MatrixXf caf_mag = caf(ref, surv,
                               n_doppler_bins, doppler_start, doppler_step,
                               sample_rate, n_range_bins);

// Find peak (target detection)
Eigen::Index doppler_bin, range_bin;
float peak_mag = caf_mag.maxCoeff(&doppler_bin, &range_bin);
```

#### CFAR Detection
```cpp
// 1D CA-CFAR
Eigen::VectorXf power_data = /* input power */;
size_t guard_cells = 4;
size_t reference_cells = 16;
float pfa_factor = 10.0f;  // Threshold multiplier

auto detections = cfar_ca(power_data, guard_cells, reference_cells, pfa_factor);
// detections[i] == 1 indicates target at index i

// 2D CFAR for range-Doppler maps
Eigen::MatrixXf range_doppler_map = /* CAF output */;
auto det_2d = cfar_2d(range_doppler_map,
                      guard_range, guard_doppler,
                      ref_range, ref_doppler,
                      pfa_factor);
```

#### Clutter Filtering
```cpp
// NLMS Adaptive Filter
Eigen::VectorXf surveillance = /* signal with clutter */;
Eigen::VectorXf reference = /* reference clutter signal */;
size_t filter_length = 64;
float mu = 0.1f;   // Adaptation rate
float eps = 1e-6f; // Regularization

Eigen::VectorXf filtered = nlms_filter(surveillance, reference,
                                        filter_length, mu, eps);

// Projection-based clutter cancellation
Eigen::MatrixXf clutter_subspace = /* basis vectors */;
Eigen::VectorXf cancelled = projection_clutter(surveillance, clutter_subspace);
```

#### Doppler Processing
```cpp
// Range-Doppler processing
Eigen::MatrixXcf pulse_data = /* n_pulses x n_range */;
size_t fft_size = 128;

Eigen::MatrixXcf doppler_map = doppler_fft(pulse_data, fft_size);

// MTI Filter (Moving Target Indicator)
Eigen::MatrixXf data = /* n_pulses x n_range */;
Eigen::VectorXf mti_coeffs(2);
mti_coeffs << 1.0f, -1.0f;  // 2-pulse canceller

Eigen::MatrixXf mti_output = mti_filter(data, mti_coeffs);
```

#### Beamforming
```cpp
// Delay-and-sum beamformer
Eigen::MatrixXf channel_data = /* n_channels x n_samples */;
Eigen::VectorXi delays = /* sample delays per channel */;
Eigen::VectorXf weights = Eigen::VectorXf::Ones(n_channels);

Eigen::VectorXf beamformed = beamform_delay_sum(channel_data, delays, weights);

// Phase-shift beamformer
Eigen::MatrixXcf complex_channels = /* n_channels x n_samples */;
Eigen::VectorXf phases = /* phase shifts in radians */;

Eigen::VectorXcf output = beamform_phase(complex_channels, phases, weights);

// Generate steering vector for Uniform Linear Array
size_t n_elements = 8;
float d_lambda = 0.5f;      // Element spacing in wavelengths
float theta = 0.3f;          // Steering angle (radians from broadside)

Eigen::VectorXcf steering = steering_vector_ula(n_elements, d_lambda, theta);
```

---

### Vulkan Kernels (`optmath::vulkan`)

#### Vector Operations
```cpp
#include <optmath/vulkan_backend.hpp>

if (optmath::vulkan::is_available()) {
    Eigen::VectorXf a = Eigen::VectorXf::Random(1000000);
    Eigen::VectorXf b = Eigen::VectorXf::Random(1000000);

    Eigen::VectorXf c = optmath::vulkan::vulkan_vec_add(a, b);
    Eigen::VectorXf d = optmath::vulkan::vulkan_vec_mul(a, b);
    float dot = optmath::vulkan::vulkan_vec_dot(a, b);
    float sum = optmath::vulkan::vulkan_reduce_sum(a);
}
```

#### Matrix Operations
```cpp
Eigen::MatrixXf A = Eigen::MatrixXf::Random(512, 512);
Eigen::MatrixXf B = Eigen::MatrixXf::Random(512, 512);

// GPU Matrix Multiplication (uses tiled shader with shared memory)
Eigen::MatrixXf C = optmath::vulkan::vulkan_mat_mul(A, B);

Eigen::MatrixXf At = optmath::vulkan::vulkan_mat_transpose(A);
```

#### FFT
```cpp
// FFT (in-place, interleaved complex format)
// Data size must be 2*N where N is power of 2
Eigen::VectorXf fft_data(2048);  // 1024 complex samples
fft_data.setRandom();

optmath::vulkan::vulkan_fft_radix2(fft_data, false);  // Forward FFT
optmath::vulkan::vulkan_fft_radix2(fft_data, true);   // Inverse FFT
```

#### Convolution
```cpp
Eigen::VectorXf signal = Eigen::VectorXf::Random(10000);
Eigen::VectorXf kernel = Eigen::VectorXf::Random(64);

Eigen::VectorXf result = optmath::vulkan::vulkan_convolution_1d(signal, kernel);

// 2D Convolution
Eigen::MatrixXf image = Eigen::MatrixXf::Random(512, 512);
Eigen::MatrixXf filter = Eigen::MatrixXf::Random(5, 5);

Eigen::MatrixXf filtered = optmath::vulkan::vulkan_convolution_2d(image, filter);
```

---

### CUDA Kernels (`optmath::cuda`)

#### Device Information
```cpp
#include <optmath/cuda_backend.hpp>

if (optmath::cuda::is_available()) {
    // Get device count
    int n_devices = optmath::cuda::get_device_count();
    std::cout << "Found " << n_devices << " CUDA device(s)\n";

    // Get detailed device info
    auto info = optmath::cuda::get_device_info(0);
    std::cout << "Device: " << info.name << "\n";
    std::cout << "Compute: " << info.compute_major() << "." << info.compute_minor() << "\n";
    std::cout << "Memory: " << info.total_memory / (1024*1024*1024) << " GB\n";
    std::cout << "Tensor Cores: " << (info.has_tensor_cores() ? "Yes" : "No") << "\n";
}
```

#### Vector Operations (cuBLAS)
```cpp
// Initialize CUDA context
optmath::cuda::init();

Eigen::VectorXf a = Eigen::VectorXf::Random(1000000);
Eigen::VectorXf b = Eigen::VectorXf::Random(1000000);

// Basic operations
Eigen::VectorXf c = optmath::cuda::cuda_add(a, b);
Eigen::VectorXf d = optmath::cuda::cuda_mul(a, b);
Eigen::VectorXf e = optmath::cuda::cuda_scale(a, 2.5f);

// Reductions
float dot = optmath::cuda::cuda_dot(a, b);
float sum = optmath::cuda::cuda_sum(a);
float max = optmath::cuda::cuda_max(a);
float min = optmath::cuda::cuda_min(a);
```

#### Matrix Operations (Tensor Core accelerated)
```cpp
Eigen::MatrixXf A = Eigen::MatrixXf::Random(1024, 1024);
Eigen::MatrixXf B = Eigen::MatrixXf::Random(1024, 1024);

// GEMM (uses Tensor Cores on Volta+)
Eigen::MatrixXf C = optmath::cuda::cuda_gemm(A, B);

// Matrix-vector multiply
Eigen::VectorXf x = Eigen::VectorXf::Random(1024);
Eigen::VectorXf y = optmath::cuda::cuda_gemv(A, x);

// Transpose
Eigen::MatrixXf At = optmath::cuda::cuda_transpose(A);
```

#### FFT (cuFFT)
```cpp
Eigen::VectorXcf signal = Eigen::VectorXcf::Random(65536);

// Forward FFT
Eigen::VectorXcf spectrum = optmath::cuda::cuda_fft(signal);

// Inverse FFT
Eigen::VectorXcf recovered = optmath::cuda::cuda_ifft(spectrum);

// 2D FFT
Eigen::MatrixXcf image = Eigen::MatrixXcf::Random(512, 512);
Eigen::MatrixXcf freq_domain = optmath::cuda::cuda_fft2(image);
```

#### Transcendental Functions (CUDA Fast Math)
```cpp
Eigen::VectorXf x = Eigen::VectorXf::Random(100000);

// Vectorized transcendentals (~10-50x faster than CPU)
Eigen::VectorXf exp_x = optmath::cuda::cuda_exp(x);
Eigen::VectorXf sin_x = optmath::cuda::cuda_sin(x);
Eigen::VectorXf cos_x = optmath::cuda::cuda_cos(x);
Eigen::VectorXf log_x = optmath::cuda::cuda_log(x.cwiseAbs());

// Activation functions (ML)
Eigen::VectorXf sig = optmath::cuda::cuda_sigmoid(x);
Eigen::VectorXf th = optmath::cuda::cuda_tanh(x);
Eigen::VectorXf relu = optmath::cuda::cuda_relu(x);
Eigen::VectorXf leaky = optmath::cuda::cuda_leaky_relu(x, 0.1f);
```

#### Complex Operations
```cpp
Eigen::VectorXcf a = Eigen::VectorXcf::Random(10000);
Eigen::VectorXcf b = Eigen::VectorXcf::Random(10000);

// Complex arithmetic
Eigen::VectorXcf prod = optmath::cuda::cuda_complex_mul(a, b);
Eigen::VectorXcf conj_prod = optmath::cuda::cuda_complex_conj_mul(a, b);  // a * conj(b)
std::complex<float> dot = optmath::cuda::cuda_complex_dot(a, b);

// Magnitude and phase
Eigen::VectorXf mag = optmath::cuda::cuda_complex_abs(a);
Eigen::VectorXf phase = optmath::cuda::cuda_complex_arg(a);
```

#### Radar Signal Processing (GPU-accelerated)
```cpp
// Cross-Ambiguity Function (CAF) - 10-50x faster than CPU
Eigen::VectorXcf ref = /* reference signal */;
Eigen::VectorXcf surv = /* surveillance signal */;

Eigen::MatrixXf caf = optmath::cuda::cuda_caf(
    ref, surv,
    64,      // n_doppler_bins
    256,     // max_range_bins
    10.0f    // doppler_step_hz
);

// 2D CFAR Detection
int guard = 2, ref_cells = 4;
float pfa = 1e-4f;
auto detections = optmath::cuda::cuda_cfar_2d(caf, guard, ref_cells, pfa);

// Bartlett Beamforming
Eigen::VectorXcf array_data = /* 4-element array */;
Eigen::VectorXf spectrum = optmath::cuda::cuda_bartlett_spectrum(
    array_data, 0.5f, 181);  // d_lambda, n_angles

// Steering Vector for ULA
Eigen::VectorXcf sv = optmath::cuda::cuda_steering_vector_ula(
    4, 0.5f, 0.523f);  // n_elements, d_lambda, angle_rad
```

#### Window Functions (GPU-generated)
```cpp
using optmath::cuda::WindowType;

// Generate windows on GPU
Eigen::VectorXf hamming = optmath::cuda::cuda_generate_window(1024, WindowType::HAMMING);
Eigen::VectorXf hanning = optmath::cuda::cuda_generate_window(1024, WindowType::HANNING);
Eigen::VectorXf blackman = optmath::cuda::cuda_generate_window(1024, WindowType::BLACKMAN);

// Apply window to data
Eigen::VectorXcf signal = /* complex signal */;
optmath::cuda::cuda_apply_window(signal, hamming);
```

#### Linear Algebra (cuSOLVER)
```cpp
Eigen::MatrixXf A = Eigen::MatrixXf::Random(256, 256);
A = A * A.transpose();  // Make symmetric positive definite

// Cholesky decomposition
Eigen::MatrixXf L = optmath::cuda::cuda_cholesky(A);

// QR decomposition
auto [Q, R] = optmath::cuda::cuda_qr(A);

// SVD
auto svd = optmath::cuda::cuda_svd(A);
// svd.U, svd.S, svd.Vt

// Linear solve Ax = b
Eigen::VectorXf b = Eigen::VectorXf::Random(256);
Eigen::VectorXf x = optmath::cuda::cuda_solve(A, b);

// Matrix inverse
Eigen::MatrixXf A_inv = optmath::cuda::cuda_inverse(A);
```

#### Multi-GPU Support
```cpp
int n_gpus = optmath::cuda::get_device_count();

for (int i = 0; i < n_gpus; ++i) {
    optmath::cuda::set_device(i);
    auto info = optmath::cuda::get_device_info(i);
    std::cout << "GPU " << i << ": " << info.name << "\n";
}

// Enable peer-to-peer access
optmath::cuda::enable_peer_access(0, 1);  // GPU 0 can access GPU 1 memory
```

#### Performance Profiling
```cpp
// CUDA event-based timing
optmath::cuda::CudaTimer timer;

timer.start();
Eigen::MatrixXf C = optmath::cuda::cuda_gemm(A, B);
optmath::cuda::synchronize();
timer.stop();

std::cout << "GEMM took " << timer.elapsed_ms() << " ms\n";

// Measure memory bandwidth
auto bw = optmath::cuda::measure_bandwidth();
std::cout << "H2D: " << bw.host_to_device_gbps << " GB/s\n";
std::cout << "D2H: " << bw.device_to_host_gbps << " GB/s\n";
```

---

## Raspberry Pi 5 Optimization Tips

1. **Vulkan Driver**: Ensure the `v3d` kernel module is loaded.
   ```bash
   lsmod | grep v3d
   ```
   If missing, add `dtoverlay=vc4-kms-v3d` to `/boot/firmware/config.txt` and reboot.

2. **Performance Guidelines**:
   - **NEON**: Faster for operations that fit in CPU cache (L2/L3) or are memory-bandwidth bound on small arrays (<100K elements).
   - **Vulkan**: Faster for heavy arithmetic (Convolution, FFT, huge Matrix Mul) where the GPU's parallelism outweighs data transfer costs.
   - **Radar Processing**: Use NEON for real-time processing; Vulkan for batch processing of large CAF computations.

3. **Cache Optimization**: The blocked GEMM is tuned for Cortex-A76:
   - L1 Data Cache: 64 KB per core
   - L2 Cache: 512 KB per core
   - Block sizes: MC=128, KC=256, NC=512

---

## Benchmarking

```bash
cmake -DBUILD_BENCHMARKS=ON ..
make -j$(nproc)

# Run individual benchmarks
./benchmarks/bench_neon_transcendentals  # exp, sin, cos, sigmoid, tanh
./benchmarks/bench_neon_gemm             # Matrix multiplication
./benchmarks/bench_neon_fft              # FIR, cross-correlation, complex ops
./benchmarks/bench_vulkan_matmul         # GPU vector/matrix operations
./benchmarks/bench_radar_caf             # CAF, CFAR, beamforming
```

Example output (Raspberry Pi 5):
```
BM_NEON_Exp_Approx/1048576    1.2 ms    1.2 ms    580  FLOPS=13.1G/s
BM_Std_Exp/1048576           45.2 ms   45.1 ms     15  FLOPS=23.2M/s
BM_NEON_ComplexMul/65536      0.1 ms    0.1 ms   6200  Elements/s=655M
BM_CAF_NEON/4096x64x256       4.8 ms    4.7 ms    148  CAF/s=212
```

---

## Environment Configuration

### Shader Location (`OPTMATH_KERNELS_PATH`)

OptMathKernels compiles shaders into SPIR-V (`.spv`) files. By default, the library looks for them in:
1. The current working directory.
2. `../src/` relative path (useful during development).
3. `/usr/local/share/optmathkernels/shaders/` (standard install path).

If you get "Shader file not found" errors:

```bash
export OPTMATH_KERNELS_PATH=/usr/local/share/optmathkernels/shaders/
./my_app
```

---

## Troubleshooting

- **"Vulkan not found" during CMake**:
  - Ensure `libvulkan-dev` is installed.
  - Check `vulkaninfo` runs correctly.

- **"Could NOT find GTest"**:
  - The build script fetches GTest automatically. Try clearing the build directory: `rm -rf build` and re-running CMake.

- **Runtime "failed to open file: vec_add.comp.spv"**:
  - Run `sudo make install` to place shaders in `/usr/local/share/...`.
  - Or set `export OPTMATH_KERNELS_PATH=/path/to/your/build/src/`.

- **NEON tests skipped**:
  - NEON is only enabled on ARM platforms. On x86, tests will skip with "NEON not available".

- **Eigen3 not found when using installed package**:
  - The CMake config includes `find_dependency(Eigen3)`. Ensure Eigen3 is installed: `sudo apt install libeigen3-dev`

---

## File Structure

```
OptMathKernels/
├── include/optmath/
│   ├── neon_kernels.hpp      # NEON API declarations
│   ├── neon_complex.hpp      # NEON complex operations
│   ├── vulkan_backend.hpp    # Vulkan API declarations
│   ├── cuda_backend.hpp      # CUDA API declarations (NEW!)
│   └── radar_kernels.hpp     # Radar processing API
├── src/
│   ├── neon/
│   │   ├── neon_kernels.cpp        # Core NEON + transcendentals
│   │   ├── neon_complex.cpp        # Complex number operations
│   │   ├── neon_gemm_optimized.cpp # Cache-blocked GEMM
│   │   └── neon_radar.cpp          # Radar signal processing
│   ├── vulkan/
│   │   ├── vulkan_backend.cpp      # Vulkan context & dispatch
│   │   └── shaders/                # 37+ GLSL compute shaders
│   └── cuda/                       # NVIDIA CUDA backend (NEW!)
│       ├── cuda_backend.cpp        # Context, memory management
│       ├── cuda_kernels.cu         # Vector ops, transcendentals
│       ├── cuda_complex.cu         # Complex ops, FFT
│       └── cuda_radar.cu           # CAF, CFAR, beamforming
├── tests/                          # GoogleTest test suites
├── benchmarks/                     # Google Benchmark suite
├── examples/
│   ├── demo.cpp                    # NEON/Vulkan demo
│   └── cuda_demo.cpp               # CUDA demo (NEW!)
└── cmake/
    └── OptMathKernelsConfig.cmake.in  # CMake package config
```

---

## Related Projects

- **[PassiveRadar_Kraken](https://github.com/n4hy/PassiveRadar_Kraken)**: Complete passive bistatic radar system using OptMathKernels for acceleration. Includes GNU Radio blocks, multi-target tracking, and real-time displays.

---

## License

MIT License - See LICENSE file for details.

---

## Author

**N4HY - Bob McGwier**
Dr Robert W McGwier, PhD
