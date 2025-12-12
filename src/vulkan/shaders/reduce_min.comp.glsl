#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Input {
    float dataIn[];
};
layout(std430, binding = 1) writeonly buffer Output {
    float dataOut[]; // Partials
};

layout(push_constant) uniform PushConsts {
    uint count;
} pc;

shared float sdata[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint idx = gl_GlobalInvocationID.x;

    // Load to shared memory
    float val = 3.402823466e+38; // FLT_MAX
    if (idx < pc.count) {
        val = dataIn[idx];
    }
    sdata[tid] = val;
    barrier();

    // Reduction in shared mem
    for (uint s = gl_WorkGroupSize.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        barrier();
    }

    if (tid == 0) {
        dataOut[gl_WorkGroupID.x] = sdata[0];
    }
}
