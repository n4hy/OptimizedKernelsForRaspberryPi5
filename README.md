# OptMathKernels

**OptMathKernels** is a high-performance C++20 numerical library optimized for **Raspberry Pi 5**. It seamlessly bridges **Eigen** (CPU), **ARM NEON** (SIMD), and **Vulkan** (Compute Shaders) into a single, easy-to-use API.

It is designed to accelerate math and signal processing tasks by leveraging the specialized hardware of the Raspberry Pi 5 (Cortex-A76 NEON and VideoCore VII GPU), while remaining compatible with standard Linux x86/ARM environments.

---

## Features

### Core Capabilities
*   **NEON Acceleration**: Hand-tuned ARMv8 NEON intrinsics for SIMD acceleration on 64-bit ARM (aarch64). Includes optimized matrix multiplication, convolution, and vector math.
*   **Vulkan Compute**: Massive parallel offloading to the GPU (VideoCore VII on Pi 5). Supports large vector operations, matrix math, FFT (Radix-2/4), and reductions.
*   **Eigen Integration**: Fully compatible with `Eigen::VectorXf`, `Eigen::MatrixXf`, and `Eigen::VectorXcf`. Pass your existing data structures directly to accelerated kernels.
*   **Easy Integration**: Standard CMake package that installs to `/usr/local` and works with `find_package(OptMathKernels)`.

### Passive Radar Signal Processing
*   **Cross-Ambiguity Function (CAF)**: Core passive radar operation for range-Doppler detection
*   **CFAR Detection**: 1D/2D Cell-Averaging and Ordered Statistic CFAR detectors
*   **Clutter Filtering**: NLMS adaptive filter and projection-based clutter cancellation
*   **Doppler Processing**: FFT-based Doppler processing and MTI filters
*   **Beamforming**: Delay-and-sum and phase-shift beamformers with steering vector generation

### Optimized Kernels
*   **Vectorized Transcendentals**: Fast NEON-accelerated exp, sin, cos, sigmoid, tanh (~10-50x faster than scalar)
*   **Complex Operations**: Vectorized complex multiply, conjugate multiply, magnitude, phase
*   **Cache-Blocked GEMM**: 8x8 microkernel with MC=128, KC=256 blocking for Cortex-A76
*   **Tiled GPU Matrix Multiply**: 16x16 shared memory tiles for efficient GPU GEMM
*   **Window Functions**: Hamming, Hanning, Blackman, Blackman-Harris, Kaiser

---

## Prerequisites

Before building, ensure you have the necessary dependencies installed.

### Raspberry Pi 5 / Ubuntu / Debian

Run the following command to install build tools, CMake, and Vulkan development headers:

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

> **Note:** GoogleTest and Google Benchmark are automatically fetched during the build (via CMake `FetchContent`).

---

## Build & Install

We provide a robust build process using CMake.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/OptMathKernels.git
cd OptMathKernels
```

### 2. Configure and Build
```bash
# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_NEON=ON \
      -DENABLE_VULKAN=ON \
      -DBUILD_TESTS=ON \
      -DBUILD_BENCHMARKS=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      ..

# Build using all available cores
make -j$(nproc)
```

### 3. Run Tests (Verification)
Verify that everything is working correctly on your hardware.
```bash
ctest --output-on-failure
```
You should see all tests pass (`test_basic`, `test_neon_kernels`, `test_radar_caf`, etc.).

### 4. Install
Install the library and headers to the system (default `/usr/local`).
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
In your project's `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyApp)

find_package(OptMathKernels REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE OptMathKernels::OptMathKernels)
```

---

## API Reference

### NEON Kernels (`optmath::neon`)

#### Vector Operations
```cpp
#include <optmath/neon_kernels.hpp>

// Check availability
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
Eigen::MatrixXf range_doppler_map = /* CAF output or similar */;
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

## Environment Configuration

### Shader Location (`OPTMATH_KERNELS_PATH`)

OptMathKernels compiles shaders into SPIR-V (`.spv`) files. By default, the library looks for them in:
1.  The current working directory.
2.  `../src/` relative path (useful during development).
3.  `/usr/local/share/optmathkernels/shaders/` (standard install path).

If you are running your application from a non-standard location and getting "Shader file not found" errors, set the `OPTMATH_KERNELS_PATH` environment variable:

```bash
export OPTMATH_KERNELS_PATH=/path/to/installation/share/optmathkernels/shaders/
./my_app
```

---

## Raspberry Pi 5 Optimization Tips

1.  **Vulkan Driver**: Ensure the `v3d` kernel module is loaded.
    ```bash
    lsmod | grep v3d
    ```
    If missing, add `dtoverlay=vc4-kms-v3d` to `/boot/firmware/config.txt` and reboot.

2.  **Performance Guidelines**:
    *   **NEON**: Faster for operations that fit in CPU cache (L2/L3) or are memory-bandwidth bound on small arrays (<100K elements).
    *   **Vulkan**: Faster for heavy arithmetic (Convolution, FFT, huge Matrix Mul) where the GPU's parallelism outweighs data transfer costs.
    *   **Radar Processing**: Use NEON for real-time processing; Vulkan for batch processing of large CAF computations.

3.  **Cache Optimization**: The blocked GEMM is tuned for Cortex-A76:
    *   L1 Data Cache: 64 KB per core
    *   L2 Cache: 512 KB per core
    *   Block sizes: MC=128, KC=256, NC=512

---

## Benchmarking

Build and run benchmarks to measure performance on your hardware:

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
```

---

## Troubleshooting

*   **"Vulkan not found" during CMake**:
    *   Ensure `libvulkan-dev` is installed.
    *   Check `vulkaninfo` runs correctly.

*   **"Could NOT find GTest"**:
    *   The build script now fetches GTest automatically. If you see issues, try clearing the build directory: `rm -rf build` and re-running CMake.

*   **Runtime "failed to open file: vec_add.comp.spv"**:
    *   This means the library cannot find the compiled shaders.
    *   **Fix**: Run `sudo make install` to place them in `/usr/local/share/...`.
    *   **Fix**: Or set `export OPTMATH_KERNELS_PATH=/path/to/your/build/src/` before running.

*   **NEON tests skipped**:
    *   NEON is only enabled on ARM platforms. On x86, tests will skip with "NEON not available".

---

## File Structure

```
OptMathKernels/
├── include/optmath/
│   ├── neon_kernels.hpp      # NEON API declarations
│   ├── vulkan_backend.hpp    # Vulkan API declarations
│   └── radar_kernels.hpp     # Radar processing API
├── src/
│   ├── neon/
│   │   ├── neon_kernels.cpp        # Core NEON + transcendentals
│   │   ├── neon_complex.cpp        # Complex number operations
│   │   ├── neon_gemm_optimized.cpp # Cache-blocked GEMM
│   │   └── neon_radar.cpp          # Radar signal processing
│   └── vulkan/
│       ├── vulkan_backend.cpp      # Vulkan context & dispatch
│       └── shaders/                # 37 GLSL compute shaders
├── tests/                          # GoogleTest test suites
├── benchmarks/                     # Google Benchmark suite
└── examples/                       # Demo applications
```

---

## License

This project is provided as-is for educational and research purposes.
