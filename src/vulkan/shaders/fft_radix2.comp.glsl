#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer Data {
    vec2 data[]; // Interleaved complex: x = real, y = imag
};

layout(push_constant) uniform PushConsts {
    uint n;      // Total size (N)
    uint stage;  // Current stage (0 to log2(N)-1)
    uint invert; // 0 = Forward, 1 = Inverse
} pc;

const float PI = 3.14159265358979323846;

void main() {
    uint tid = gl_GlobalInvocationID.x;
    // Radix-2 Butterfly (Stockham or Cooley-Tukey)
    // Here we implement a standard butterfly for bit-reversed inputs or standard iterative?
    // Let's implement Stockham (no bit reversal needed, but ping-pong buffers usually).
    // Or iterative Cooley-Tukey (in-place) which requires bit-reversal first?
    // Given we want a single kernel call per stage or similar.

    // Simplest for single shader file to be called multiple times:
    // "Cooley-Tukey Butterfly"
    // For stage s (0..logN-1):
    // Span = 2^s
    // Group = tid / Span
    // Offset = tid % Span
    // Pair = Group * 2 * Span + Offset
    // Left = Pair
    // Right = Pair + Span

    // But we need N/2 threads.

    uint N = pc.n;
    uint p = 1 << pc.stage; // Span
    uint k = tid & (p - 1); // Index inside butterfly group
    uint j = ((tid - k) << 1) + k; // Top index

    // j is the index of the "top" element of the butterfly
    // j + p is the "bottom"

    if (j + p < N) {
        vec2 u = data[j];
        vec2 v = data[j + p];

        // Twiddle factor
        float angle = -2.0 * PI * float(k) / float(2 * p);
        if (pc.invert != 0) angle = -angle;

        vec2 w = vec2(cos(angle), sin(angle));

        // Complex mul v * w
        vec2 vw = vec2(v.x * w.x - v.y * w.y, v.x * w.y + v.y * w.x);

        data[j] = u + vw;
        data[j + p] = u - vw;
    }
}
