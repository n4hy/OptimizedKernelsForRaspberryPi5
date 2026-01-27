#version 450
// Phase 2 of Multi-Block Scan: Scan Block Sums
// Performs inclusive prefix sum on the block sums from phase 1
// This creates the offsets needed to complete the global scan

layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer BlockSums {
    float blockSums[];
};

layout(push_constant) uniform PushConsts {
    uint numBlocks; // Number of blocks to scan
} pc;

shared float sdata[512];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint blockDim = gl_WorkGroupSize.x;

    // This shader handles scanning up to 512 block sums
    // For larger arrays, this would need to be called recursively

    // Load block sums (two per thread)
    uint ai = tid;
    uint bi = tid + blockDim;

    sdata[ai] = (ai < pc.numBlocks) ? blockSums[ai] : 0.0;
    sdata[bi] = (bi < pc.numBlocks) ? blockSums[bi] : 0.0;

    // Up-sweep (reduce) phase
    uint offset = 1;
    for (uint d = blockDim; d > 0; d >>= 1) {
        barrier();
        if (tid < d) {
            uint aiOffset = offset * (2 * tid + 1) - 1;
            uint biOffset = offset * (2 * tid + 2) - 1;
            if (biOffset < 2 * blockDim) {
                sdata[biOffset] += sdata[aiOffset];
            }
        }
        offset *= 2;
    }

    // Clear the last element for exclusive scan
    if (tid == 0) {
        sdata[2 * blockDim - 1] = 0.0;
    }

    // Down-sweep phase
    for (uint d = 1; d < 2 * blockDim; d *= 2) {
        offset >>= 1;
        barrier();
        if (tid < d) {
            uint aiOffset = offset * (2 * tid + 1) - 1;
            uint biOffset = offset * (2 * tid + 2) - 1;
            if (biOffset < 2 * blockDim) {
                float t = sdata[aiOffset];
                sdata[aiOffset] = sdata[biOffset];
                sdata[biOffset] += t;
            }
        }
    }
    barrier();

    // Write back scanned block sums (these are now prefix sums / offsets)
    // We write exclusive scan results - the offset for each block
    if (ai < pc.numBlocks) {
        blockSums[ai] = sdata[ai];
    }
    if (bi < pc.numBlocks) {
        blockSums[bi] = sdata[bi];
    }
}
