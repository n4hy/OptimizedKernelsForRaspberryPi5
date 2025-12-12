#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer InputX {
    float dataX[];
};
layout(std430, binding = 1) readonly buffer InputH {
    float dataH[];
};
layout(std430, binding = 2) writeonly buffer Output {
    float dataY[];
};

layout(push_constant) uniform PushConsts {
    uint n_x;
    uint n_h;
} pc;

void main() {
    // Correlation: (f star g)[n] = sum_m f[n+m] g[m] (complex conjugate if complex)
    // Here reals.
    // Valid correlation size: n_x - n_h + 1.

    uint n = gl_GlobalInvocationID.x;
    uint n_out = pc.n_x - pc.n_h + 1;

    if (n < n_out) {
        float sum = 0.0;
        for (uint k = 0; k < pc.n_h; ++k) {
            sum += dataX[n + k] * dataH[k];
        }
        dataY[n] = sum;
    }
}
