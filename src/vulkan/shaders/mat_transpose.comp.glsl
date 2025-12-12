#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer InputA {
    float dataA[];
};
layout(std430, binding = 1) writeonly buffer Output {
    float dataOut[];
};

layout(push_constant) uniform PushConsts {
    uint rows;
    uint cols;
} pc;

void main() {
    uint r = gl_GlobalInvocationID.x; // row index in A
    uint c = gl_GlobalInvocationID.y; // col index in A

    if (r < pc.rows && c < pc.cols) {
        // A is rows x cols.
        // A[r, c] -> idx = c * rows + r (col-major)
        float val = dataA[c * pc.rows + r];

        // Output is cols x rows.
        // Out[c, r] -> idx = r * cols + c (col-major)
        dataOut[r * pc.cols + c] = val;
    }
}
