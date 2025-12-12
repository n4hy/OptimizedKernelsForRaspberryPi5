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

vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

void main() {
    uint tid = gl_GlobalInvocationID.x;
    uint N = pc.n;
    uint stride = 1 << (2 * pc.stage);
    uint group = tid / stride;
    uint offset = tid % stride;

    uint base = group * 4 * stride + offset;

    if (base + 3 * stride < N) {
        vec2 v0 = data[base];
        vec2 v1 = data[base + stride];
        vec2 v2 = data[base + 2 * stride];
        vec2 v3 = data[base + 3 * stride];

        float L = float(4 * stride);
        // Inverse: + angle
        float angle = TWO_PI * float(offset) / L;

        vec2 w1 = vec2(cos(angle), sin(angle));
        vec2 w2 = cmul(w1, w1);
        vec2 w3 = cmul(w1, w2);

        v1 = cmul(v1, w1);
        v2 = cmul(v2, w2);
        v3 = cmul(v3, w3);

        vec2 t0 = v0 + v2;
        vec2 t1 = v0 - v2;
        vec2 t2 = v1 + v3;
        vec2 t3 = v1 - v3;

        vec2 out0 = t0 + t2;
        vec2 out2 = t0 - t2;

        vec2 out1, out3;
        // Inverse: +i
        out1 = vec2(t1.x - t3.y, t1.y + t3.x);
        out3 = vec2(t1.x + t3.y, t1.y - t3.x);

        data[base] = out0;
        data[base + stride] = out1;
        data[base + 2 * stride] = out2;
        data[base + 3 * stride] = out3;
    }
}
