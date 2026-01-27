#version 450
// Phase 3 of Multi-Block Scan: Add Block Offsets
// Adds the scanned block sums to each element in the corresponding block
// Completes the global prefix sum

layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer Data {
    float data[];
};
layout(std430, binding = 1) readonly buffer BlockSums {
    float blockSums[]; // Exclusive prefix sum of block sums
};

layout(push_constant) uniform PushConsts {
    uint count;
    uint elementsPerBlock; // = 2 * workgroup_size from phase 1
} pc;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint blockIdx = gl_WorkGroupID.x;

    if (gid >= pc.count) return;

    // Get the offset for this block (from the scanned block sums)
    float offset = blockSums[blockIdx];

    // Add offset to each element
    data[gid] += offset;
}
