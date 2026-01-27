#version 450
layout(local_size_x = 256) in;

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
    uint count;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.count) {
        // Add epsilon to avoid division by zero
        float denom = dataB[idx];
        float epsilon = 1e-10;
        // Sign-preserving epsilon: prevents zero without changing sign
        if (abs(denom) < epsilon) {
            denom = (denom >= 0.0) ? epsilon : -epsilon;
        }
        dataOut[idx] = dataA[idx] / denom;
    }
}
