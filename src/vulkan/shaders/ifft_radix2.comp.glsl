#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer Data {
    vec2 data[];
};

layout(push_constant) uniform PushConsts {
    uint n;
    uint stage;
    uint invert;
} pc;

const float PI = 3.14159265358979323846;

void main() {
    uint tid = gl_GlobalInvocationID.x;
    // Same as forward, but we just flip the angle sign in logic.
    // In strict sense, ifft_radix2.comp might just hardcode invert=1
    // Reusing the logic:

    uint N = pc.n;
    uint p = 1 << pc.stage;
    uint k = tid & (p - 1);
    uint j = ((tid - k) << 1) + k;

    if (j + p < N) {
        vec2 u = data[j];
        vec2 v = data[j + p];

        // Twiddle factor (Positive angle for IFFT)
        float angle = 2.0 * PI * float(k) / float(2 * p);

        vec2 w = vec2(cos(angle), sin(angle));

        vec2 vw = vec2(v.x * w.x - v.y * w.y, v.x * w.y + v.y * w.x);

        data[j] = u + vw;
        data[j + p] = u - vw;
    }
}
