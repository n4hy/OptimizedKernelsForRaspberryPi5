#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Input {
    float dataIn[];
};
layout(std430, binding = 1) writeonly buffer Output {
    float dataOut[];
};

layout(push_constant) uniform PushConsts {
    uint count;
} pc;

shared float temp[512]; // 2 * local_size

void main() {
    uint thid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;

    // Load input into shared memory
    // Exclusive scan: shift right.
    // Hillis-Steele or Blelloch.
    // Let's do a simple workgroup scan (Blelloch)

    // 1. Load
    uint n = 256;
    // Each thread loads 2 elements? No, simpler to just map 1:1 if possible.
    // Let's assume input size <= 256 for this basic kernel, or it just scans the block.

    float val = 0.0;
    if (gid < pc.count) val = dataIn[gid];
    temp[thid] = val;
    barrier();

    // 2. Up-sweep (Reduce)
    uint offset = 1;
    for (uint d = n >> 1; d > 0; d >>= 1) {
        if (thid < d) {
            uint ai = offset * (2 * thid + 1) - 1;
            uint bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
        barrier();
    }

    // 3. Clear last element
    if (thid == 0) temp[n - 1] = 0;
    barrier();

    // 4. Down-sweep
    for (uint d = 1; d < n; d *= 2) {
        offset >>= 1;
        if (thid < d) {
            uint ai = offset * (2 * thid + 1) - 1;
            uint bi = offset * (2 * thid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        barrier();
    }

    // Write out
    if (gid < pc.count) {
        dataOut[gid] = temp[thid]; // Exclusive scan
        // If inclusive needed: dataOut[gid] = temp[thid] + val;
    }
}
