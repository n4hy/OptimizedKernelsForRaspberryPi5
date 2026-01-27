#version 450
// Optimized 2D Convolution with Kernel in Shared Memory
// Uses tiled input loading with halo regions for boundary handling

layout(local_size_x = 16, local_size_y = 16) in;

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
    uint inputRows;
    uint inputCols;
    uint kernelRows;
    uint kernelCols;
    uint outputRows;
    uint outputCols;
} pc;

// Maximum kernel dimensions
#define MAX_KERNEL_DIM 15
#define TILE_SIZE 16
#define MAX_TILE_WITH_HALO (TILE_SIZE + MAX_KERNEL_DIM)

// Shared memory
shared float sharedKernel[MAX_KERNEL_DIM][MAX_KERNEL_DIM];
shared float sharedInput[MAX_TILE_WITH_HALO][MAX_TILE_WITH_HALO];

void main() {
    uint localRow = gl_LocalInvocationID.x;
    uint localCol = gl_LocalInvocationID.y;
    uint globalRow = gl_GlobalInvocationID.x;
    uint globalCol = gl_GlobalInvocationID.y;

    uint blockStartRow = gl_WorkGroupID.x * TILE_SIZE;
    uint blockStartCol = gl_WorkGroupID.y * TILE_SIZE;

    int halfKernelRow = int(pc.kernelRows) / 2;
    int halfKernelCol = int(pc.kernelCols) / 2;

    // Load kernel into shared memory (cooperatively)
    // Each thread in the 16x16 block loads part of the kernel
    if (localRow < pc.kernelRows && localCol < pc.kernelCols) {
        sharedKernel[localRow][localCol] = kernel[localRow * pc.kernelCols + localCol];
    }

    // Calculate tile dimensions with halo
    uint tileSizeWithHaloRow = TILE_SIZE + pc.kernelRows - 1;
    uint tileSizeWithHaloCol = TILE_SIZE + pc.kernelCols - 1;

    // Load input tile with halo into shared memory
    // Each thread may need to load multiple elements
    for (uint r = localRow; r < tileSizeWithHaloRow; r += TILE_SIZE) {
        for (uint c = localCol; c < tileSizeWithHaloCol; c += TILE_SIZE) {
            int inputRow = int(blockStartRow) + int(r) - halfKernelRow;
            int inputCol = int(blockStartCol) + int(c) - halfKernelCol;

            if (inputRow >= 0 && inputRow < int(pc.inputRows) &&
                inputCol >= 0 && inputCol < int(pc.inputCols)) {
                sharedInput[r][c] = dataIn[inputRow * pc.inputCols + inputCol];
            } else {
                sharedInput[r][c] = 0.0; // Zero padding
            }
        }
    }

    barrier();

    // Compute convolution for this output element
    if (globalRow < pc.outputRows && globalCol < pc.outputCols) {
        float sum = 0.0;

        for (uint kr = 0; kr < pc.kernelRows; ++kr) {
            for (uint kc = 0; kc < pc.kernelCols; ++kc) {
                sum += sharedInput[localRow + kr][localCol + kc] * sharedKernel[kr][kc];
            }
        }

        dataOut[globalRow * pc.outputCols + globalCol] = sum;
    }
}
