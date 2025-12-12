#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer InputA {
    float dataA[];
};
layout(std430, binding = 1) readonly buffer InputB {
    float dataB[];
};
layout(std430, binding = 2) writeonly buffer Output {
    float dataOut[];
};

layout(push_constant) uniform PushConsts {
    uint rows;
    uint cols;
} pc;

void main() {
    uint r = gl_GlobalInvocationID.x;
    uint c = gl_GlobalInvocationID.y;

    if (r < pc.rows && c < pc.cols) {
        // Column-major indexing
        uint idx = c * pc.rows + r;
        dataOut[idx] = dataA[idx] + dataB[idx];
    }
}
