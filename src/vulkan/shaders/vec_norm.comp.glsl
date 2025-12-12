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
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.count) {
        float v = dataA[idx];
        // Just square elements, we will reduce on CPU or another pass.
        // Or wait, vec_norm typically means Euclidean norm ||x||_2 = sqrt(sum(x^2))
        // This shader will just do the element-wise square. The summation and sqrt happens on CPU for now (consistent with vec_dot).
        dataOut[idx] = v * v;
    }
}
