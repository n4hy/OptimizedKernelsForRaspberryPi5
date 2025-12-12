#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer Data {
    vec2 data[];
};

layout(push_constant) uniform PushConsts {
    uint n;
    uint stage; // Power of 4 stage
    uint invert;
} pc;

const float PI = 3.14159265358979323846;
const float TWO_PI = 6.2831853071795864769;

// Complex Mul
vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

void main() {
    uint tid = gl_GlobalInvocationID.x;
    uint N = pc.n;

    // 4 elements per butterfly
    // Threads = N / 4

    uint p = 1 << (2 * pc.stage); // 4^stage -> current group size? No.
    // Length of sub-transform L = 4^(stage+1).
    // Let L = 4 * p.
    // p is stride?
    // Let's align with radix-2 logic.
    // Stage 0: groups of 4. Stride 1.
    // Stage 1: groups of 16. Stride 4.

    uint stride = 1 << (2 * pc.stage); // 1, 4, 16...
    uint group = tid / stride;
    uint offset = tid % stride;

    uint base = group * 4 * stride + offset;

    if (base + 3 * stride < N) {
        vec2 v0 = data[base];
        vec2 v1 = data[base + stride];
        vec2 v2 = data[base + 2 * stride];
        vec2 v3 = data[base + 3 * stride];

        // Twiddle factors
        // W_N^k = exp(-i 2pi k / N)
        // Here N is actually current length L = 4*stride.
        // k = offset.
        // w1 = W_L^k
        // w2 = W_L^{2k}
        // w3 = W_L^{3k}

        float L = float(4 * stride);
        float angle = -TWO_PI * float(offset) / L;
        if (pc.invert != 0) angle = -angle;

        vec2 w1 = vec2(cos(angle), sin(angle));
        vec2 w2 = cmul(w1, w1);
        vec2 w3 = cmul(w1, w2);

        // Apply twiddles
        v1 = cmul(v1, w1);
        v2 = cmul(v2, w2);
        v3 = cmul(v3, w3);

        // Radix-4 Butterfly (DIT)
        // t0 = v0 + v2
        // t1 = v0 - v2
        // t2 = v1 + v3
        // t3 = v1 - v3

        // out0 = t0 + t2
        // out1 = t1 - i*t3 (if fwd)
        // out2 = t0 - t2
        // out3 = t1 + i*t3 (if fwd)

        vec2 t0 = v0 + v2;
        vec2 t1 = v0 - v2;
        vec2 t2 = v1 + v3;
        vec2 t3 = v1 - v3;

        vec2 out0 = t0 + t2;
        vec2 out2 = t0 - t2;

        vec2 out1, out3;
        if (pc.invert != 0) {
            // Inverse: +i
            // t1 + i*t3 = (t1.x - t3.y, t1.y + t3.x)
            out1 = vec2(t1.x - t3.y, t1.y + t3.x);
            // t1 - i*t3
            out3 = vec2(t1.x + t3.y, t1.y - t3.x);
        } else {
            // Forward: -i
            // t1 - i*t3 = (t1.x + t3.y, t1.y - t3.x)
            out1 = vec2(t1.x + t3.y, t1.y - t3.x);
            // t1 + i*t3
            out3 = vec2(t1.x - t3.y, t1.y + t3.x);
        }

        data[base] = out0;
        data[base + stride] = out1;
        data[base + 2 * stride] = out2;
        data[base + 3 * stride] = out3;
    }
}
