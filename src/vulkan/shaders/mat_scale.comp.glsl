#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer InputA {
    float dataA[];
};
layout(std430, binding = 1) writeonly buffer Output {
    float dataOut[];
};

layout(push_constant) uniform PushConsts {
    uint count;
    float scalar;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.count) {
        dataOut[idx] = dataA[idx] * pc.scalar;
    }
}
