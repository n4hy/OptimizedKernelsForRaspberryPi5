#version 450
layout(local_size_x = 16, local_size_y = 16) in;

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
    uint M; // Rows of A
    uint K; // Cols of A = Rows of B
    uint N; // Cols of B
} pc;

void main() {
    // Result C is M x N
    uint row = gl_GlobalInvocationID.x; // 0..M-1
    uint col = gl_GlobalInvocationID.y; // 0..N-1

    if (row < pc.M && col < pc.N) {
        float sum = 0.0;
        for (uint k = 0; k < pc.K; ++k) {
            // A is M x K (col-major): A[row, k] -> index = k * M + row
            float valA = dataA[k * pc.M + row];
            // B is K x N (col-major): B[k, col] -> index = col * K + k
            float valB = dataB[col * pc.K + k];
            sum += valA * valB;
        }
        // C is M x N (col-major): C[row, col] -> index = col * M + row
        dataOut[col * pc.M + row] = sum;
    }
}
