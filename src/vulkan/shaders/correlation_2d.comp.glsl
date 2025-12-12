#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer InputX {
    float dataX[];
};
layout(std430, binding = 1) readonly buffer InputK {
    float dataK[];
};
layout(std430, binding = 2) writeonly buffer Output {
    float dataY[];
};

layout(push_constant) uniform PushConsts {
    uint H_in;
    uint W_in;
    uint K_h;
    uint K_w;
} pc;

void main() {
    uint H_out = pc.H_in - pc.K_h + 1;
    uint W_out = pc.W_in - pc.K_w + 1;

    uint r = gl_GlobalInvocationID.x;
    uint c = gl_GlobalInvocationID.y;

    if (r < H_out && c < W_out) {
        float sum = 0.0;
        // Correlation: sum_{i,j} Input[r+i, c+j] * Kernel[i, j] (No flip)

        for (uint ki = 0; ki < pc.K_h; ++ki) {
            for (uint kj = 0; kj < pc.K_w; ++kj) {
                // Col-major: idx = col * Height + row
                float valInput = dataX[(c + kj) * pc.H_in + (r + ki)];
                float valKernel = dataK[kj * pc.K_h + ki];
                sum += valInput * valKernel;
            }
        }
        dataY[c * H_out + r] = sum;
    }
}
