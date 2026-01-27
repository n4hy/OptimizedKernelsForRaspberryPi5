#version 450
// Phase 1 of Multi-Block Scan: Local Workgroup Scan
// Performs inclusive prefix sum within each workgroup
// Outputs both the scanned data and the block sums

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Input {
    float dataIn[];
};
layout(std430, binding = 1) writeonly buffer Output {
    float dataOut[];
};
layout(std430, binding = 2) writeonly buffer BlockSums {
    float blockSums[]; // Sum of each workgroup
};

layout(push_constant) uniform PushConsts {
    uint count;
} pc;

shared float sdata[512]; // Double size for work-efficient algorithm

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;
    uint blockIdx = gl_WorkGroupID.x;
    uint blockDim = gl_WorkGroupSize.x;

    // Load data into shared memory (two elements per thread for work-efficient scan)
    uint ai = tid;
    uint bi = tid + blockDim;

    // Global indices
    uint globalAi = 2 * blockIdx * blockDim + ai;
    uint globalBi = 2 * blockIdx * blockDim + bi;

    sdata[ai] = (globalAi < pc.count) ? dataIn[globalAi] : 0.0;
    sdata[bi] = (globalBi < pc.count) ? dataIn[globalBi] : 0.0;

    // Build the sum in place (up-sweep / reduce phase)
    uint offset = 1;
    for (uint d = blockDim; d > 0; d >>= 1) {
        barrier();
        if (tid < d) {
            uint aiOffset = offset * (2 * tid + 1) - 1;
            uint biOffset = offset * (2 * tid + 2) - 1;
            sdata[biOffset] += sdata[aiOffset];
        }
        offset *= 2;
    }

    // Store the block sum and clear the last element
    if (tid == 0) {
        blockSums[blockIdx] = sdata[2 * blockDim - 1];
        sdata[2 * blockDim - 1] = 0.0;
    }

    // Down-sweep phase
    for (uint d = 1; d < 2 * blockDim; d *= 2) {
        offset >>= 1;
        barrier();
        if (tid < d) {
            uint aiOffset = offset * (2 * tid + 1) - 1;
            uint biOffset = offset * (2 * tid + 2) - 1;
            float t = sdata[aiOffset];
            sdata[aiOffset] = sdata[biOffset];
            sdata[biOffset] += t;
        }
    }
    barrier();

    // Write results (convert exclusive scan to inclusive by shifting)
    if (globalAi < pc.count) {
        dataOut[globalAi] = sdata[ai] + dataIn[globalAi];
    }
    if (globalBi < pc.count) {
        dataOut[globalBi] = sdata[bi] + dataIn[globalBi];
    }
}
