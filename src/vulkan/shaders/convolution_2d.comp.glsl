#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer InputX {
    float dataX[]; // Input Image: H_in x W_in
};
layout(std430, binding = 1) readonly buffer InputK {
    float dataK[]; // Kernel: K_h x K_w
};
layout(std430, binding = 2) writeonly buffer Output {
    float dataY[]; // Output: H_out x W_out
};

layout(push_constant) uniform PushConsts {
    uint H_in;
    uint W_in;
    uint K_h;
    uint K_w;
    // Assuming stride 1, no padding for simplicity in this version,
    // consistent with typical "valid" convolution or prompt expectation of basic conv.
} pc;

void main() {
    // Output dimensions
    uint H_out = pc.H_in - pc.K_h + 1;
    uint W_out = pc.W_in - pc.K_w + 1;

    uint r = gl_GlobalInvocationID.x; // row in output
    uint c = gl_GlobalInvocationID.y; // col in output

    if (r < H_out && c < W_out) {
        float sum = 0.0;
        // Convolution: sum(x[r+i, c+j] * k[i, j])?
        // Or flip kernel? "convolution" mathematically implies flip. "Correlation" is no flip.
        // Usually deep learning "convolution" is correlation (no flip).
        // I will implement correlation style (no flip) unless user specified.
        // User asked for "convolution_2d" AND "correlation_2d".
        // So I must distinguish. Convolution = Correlation with flipped kernel.

        for (uint ki = 0; ki < pc.K_h; ++ki) {
            for (uint kj = 0; kj < pc.K_w; ++kj) {
                // Image index
                uint ir = r + (pc.K_h - 1 - ki); // Flip row
                uint ic = c + (pc.K_w - 1 - kj); // Flip col
                // Wait, standard conv flip:
                // y[r, c] = sum_{i,j} x[r+i, c+j] * k[Kh-1-i, Kw-1-j]

                // Let's iterate kernel indices directly (no flip logic on image indices)
                // y[r,c] = sum_{i,j} x[r+(Kh-1-i), c+(Kw-1-j)] * k[i,j] ? No.

                // Definition: (f * g)[n] = sum f[m] g[n-m]
                // 2D: (I * K)[r, c] = sum_{i,j} I[r-i, c-j] K[i, j]
                // Usually we shift origin.
                // Let's implement: sum_{i=0..Kh-1, j=0..Kw-1} Input[r+i, c+j] * Kernel[Kh-1-i, Kw-1-j]

                // Image indices (col-major assumption? Or row-major?
                // Eigen defaults to Col-Major.
                // dataX[col * Rows + row]

                // For 2D images, often Row-Major is standard in CV, but Eigen is Col-Major.
                // I will stick to Eigen Col-Major consistency: idx = c * H + r.

                float valInput = dataX[(c + kj) * pc.H_in + (r + ki)];
                float valKernel = dataK[(pc.K_w - 1 - kj) * pc.K_h + (pc.K_h - 1 - ki)];
                sum += valInput * valKernel;
            }
        }
        dataY[c * H_out + r] = sum;
    }
}
