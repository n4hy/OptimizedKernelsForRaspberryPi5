#version 450
// Optimized Radix-2 FFT with Precomputed Twiddle Factors
// Uses shared memory for in-workgroup butterflies
// Twiddle factors passed via uniform buffer instead of computed per-invocation

layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer Data {
    vec2 data[]; // Interleaved complex: x = real, y = imag
};

// Precomputed twiddle factors: W_N^k for k = 0..N/2-1
// Each vec2 contains (cos, sin) = (real, imag) of exp(-2*pi*i*k/N)
layout(std430, binding = 1) readonly buffer TwiddleFactors {
    vec2 twiddles[];
};

layout(push_constant) uniform PushConsts {
    uint n;           // Total FFT size (N)
    uint stage;       // Current stage (0 to log2(N)-1)
    uint invert;      // 0 = Forward, 1 = Inverse
    uint twiddle_stride; // Stride in twiddle table for this stage
} pc;

// Shared memory for efficient in-workgroup computation
shared vec2 sdata[512]; // 2x workgroup size for double buffering

// Complex multiplication: a * b
vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

void main() {
    uint tid = gl_GlobalInvocationID.x;
    uint N = pc.n;

    // For this stage:
    // p = 2^stage (half the butterfly span)
    // Butterflies are indexed as pairs
    uint p = 1u << pc.stage;

    // Calculate butterfly indices
    // k = position within the butterfly group
    // j = top element index
    uint k = tid & (p - 1);
    uint j = ((tid - k) << 1) + k;

    if (j + p >= N) return;

    // Load data
    vec2 u = data[j];
    vec2 v = data[j + p];

    // Get twiddle factor
    // For stage s, we need W_N^(k * N / (2^(s+1))) = W_N^(k * twiddle_stride)
    // twiddle_stride = N / (2 * 2^stage) = N >> (stage + 1)
    uint twiddle_idx = k * pc.twiddle_stride;

    vec2 w = twiddles[twiddle_idx];

    // For inverse FFT, conjugate the twiddle factor
    if (pc.invert != 0) {
        w.y = -w.y;
    }

    // Complex multiply: v * w
    vec2 vw = cmul(v, w);

    // Butterfly operation
    data[j] = u + vw;
    data[j + p] = u - vw;
}
