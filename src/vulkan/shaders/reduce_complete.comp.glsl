#version 450
// Complete GPU-side Reduction with Atomic Finalization
// Single kernel that completes the entire reduction without CPU intervention
// Uses hierarchical workgroup reduction followed by atomic accumulation

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Input {
    float dataIn[];
};
layout(std430, binding = 1) buffer Output {
    uint result_bits;  // Result stored as uint bits for atomic CAS
    uint counter;      // Atomic counter for workgroup completion
};

layout(push_constant) uniform PushConsts {
    uint count;         // Total number of elements
    uint num_workgroups; // Total number of workgroups
} pc;

shared float sdata[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;
    uint workgroupSize = gl_WorkGroupSize.x;

    // Load and accumulate multiple elements per thread for better efficiency
    // Each thread handles multiple elements with stride = total threads
    float sum = 0.0;
    uint stride = gl_NumWorkGroups.x * workgroupSize;

    for (uint i = gid; i < pc.count; i += stride) {
        sum += dataIn[i];
    }

    sdata[tid] = sum;
    barrier();

    // Reduction in shared memory
    for (uint s = workgroupSize / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier();
    }

    // Warp-level reduction (no barrier needed for warp-synchronous execution)
    // Note: Vulkan doesn't guarantee warp-synchronous, but subgroup ops can help
    // For safety, we continue with barriers
    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
        barrier();
        sdata[tid] += sdata[tid + 16];
        barrier();
        sdata[tid] += sdata[tid + 8];
        barrier();
        sdata[tid] += sdata[tid + 4];
        barrier();
        sdata[tid] += sdata[tid + 2];
        barrier();
        sdata[tid] += sdata[tid + 1];
    }
    barrier();

    // Thread 0 of each workgroup atomically adds to global result
    if (tid == 0) {
        // Atomic add for float (requires VK_EXT_shader_atomic_float or manual CAS)
        // Using atomicAdd assuming the extension is available
        // If not available, this shader needs modification to use integer atomics

        // For compatibility, we'll use a two-phase approach:
        // 1. Store workgroup partial in a temporary array
        // 2. Last workgroup to finish sums all partials

        // Atomic increment counter
        uint oldCount = atomicAdd(counter, 1u);

        // Store our partial sum (we need a partials buffer for this)
        // For simplicity in this version, we use atomicAdd on result
        // This works with VK_EXT_shader_atomic_float

        // Fallback: Use integer atomics by reinterpreting float bits
        // This is a simplified version - production code should use proper atomic float

        // Atomic add emulation using integer CAS on result_bits buffer
        uint old_bits = result_bits;
        uint new_bits;
        float old_val, new_val;

        do {
            old_val = uintBitsToFloat(old_bits);
            new_val = old_val + sdata[0];
            new_bits = floatBitsToUint(new_val);
            // atomicCompSwap returns the original value at the memory location
            uint actual_old = atomicCompSwap(result_bits, old_bits, new_bits);
            if (actual_old == old_bits) {
                break;  // Success
            }
            old_bits = actual_old;  // Retry with actual value
        } while (true);
    }
}
