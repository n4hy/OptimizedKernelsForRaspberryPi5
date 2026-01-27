#version 450
// Optimized 1D Convolution with Kernel in Shared Memory
// Caches the convolution kernel in shared memory for reuse
// Uses tiled input loading with halo regions

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Input {
    float dataIn[];
};
layout(std430, binding = 1) readonly buffer Kernel {
    float kernel[];
};
layout(std430, binding = 2) writeonly buffer Output {
    float dataOut[];
};

layout(push_constant) uniform PushConsts {
    uint inputSize;
    uint kernelSize;
    uint outputSize;
} pc;

// Maximum kernel size we support in shared memory
#define MAX_KERNEL_SIZE 128

// Shared memory for kernel and input tile with halo
shared float sharedKernel[MAX_KERNEL_SIZE];
shared float sharedInput[256 + MAX_KERNEL_SIZE]; // Tile + halo

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;
    uint blockSize = gl_WorkGroupSize.x;
    uint blockStart = gl_WorkGroupID.x * blockSize;

    // Load kernel into shared memory (cooperatively)
    // Each thread loads one or more kernel elements
    for (uint i = tid; i < pc.kernelSize && i < MAX_KERNEL_SIZE; i += blockSize) {
        sharedKernel[i] = kernel[i];
    }

    // Calculate halo size
    uint halfKernel = pc.kernelSize / 2;
    uint tileSize = blockSize;
    uint totalLoadSize = tileSize + pc.kernelSize - 1;

    // Load input tile with halo into shared memory
    // Main tile region
    int inputIdx = int(blockStart) + int(tid) - int(halfKernel);
    if (tid < totalLoadSize) {
        if (inputIdx >= 0 && inputIdx < int(pc.inputSize)) {
            sharedInput[tid] = dataIn[inputIdx];
        } else {
            sharedInput[tid] = 0.0; // Zero padding for boundaries
        }
    }

    // Load additional elements if tile + halo > blockSize
    if (tid + blockSize < totalLoadSize) {
        int extraIdx = int(blockStart) + int(tid + blockSize) - int(halfKernel);
        if (extraIdx >= 0 && extraIdx < int(pc.inputSize)) {
            sharedInput[tid + blockSize] = dataIn[extraIdx];
        } else {
            sharedInput[tid + blockSize] = 0.0;
        }
    }

    barrier();

    // Compute convolution for this output element
    if (gid < pc.outputSize) {
        float sum = 0.0;

        // The output at position gid corresponds to centering the kernel at input[gid]
        // In shared memory, our local position is at sharedInput[tid + halfKernel] for centered
        // But we load with halo, so position 0 in shared is input[blockStart - halfKernel]
        // Output[gid] = sum over k: input[gid - halfKernel + k] * kernel[k]
        // = sum over k: sharedInput[tid + k] * sharedKernel[k]

        for (uint k = 0; k < pc.kernelSize; ++k) {
            sum += sharedInput[tid + k] * sharedKernel[k];
        }

        dataOut[gid] = sum;
    }
}
