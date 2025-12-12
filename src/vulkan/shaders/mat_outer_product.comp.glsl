#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer InputVecU {
    float dataU[]; // M
};
layout(std430, binding = 1) readonly buffer InputVecV {
    float dataV[]; // N
};
layout(std430, binding = 2) writeonly buffer Output {
    float dataOut[]; // MxN
};

layout(push_constant) uniform PushConsts {
    uint rows; // M
    uint cols; // N
} pc;

void main() {
    uint r = gl_GlobalInvocationID.x;
    uint c = gl_GlobalInvocationID.y;

    if (r < pc.rows && c < pc.cols) {
        // Outer product: A = u * v^T
        // A[r, c] = u[r] * v[c]
        // Output col-major: idx = c*rows + r
        dataOut[c * pc.rows + r] = dataU[r] * dataV[c];
    }
}
