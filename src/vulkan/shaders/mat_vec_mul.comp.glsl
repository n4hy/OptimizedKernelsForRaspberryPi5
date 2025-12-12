#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer InputMat {
    float dataMat[]; // MxN
};
layout(std430, binding = 1) readonly buffer InputVec {
    float dataVec[]; // N
};
layout(std430, binding = 2) writeonly buffer Output {
    float dataOut[]; // M
};

layout(push_constant) uniform PushConsts {
    uint rows; // M
    uint cols; // N
} pc;

void main() {
    uint r = gl_GlobalInvocationID.x;
    if (r < pc.rows) {
        float sum = 0.0;
        for (uint c = 0; c < pc.cols; ++c) {
            // Mat is col-major: M[r, c] -> idx = c*rows + r
            float valM = dataMat[c * pc.rows + r];
            float valV = dataVec[c];
            sum += valM * valV;
        }
        dataOut[r] = sum;
    }
}
