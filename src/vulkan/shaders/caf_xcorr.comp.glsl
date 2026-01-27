#version 450
// Cross-Ambiguity Function - Phase 2: Cross-Correlation
// Computes correlation between Doppler-shifted reference and surveillance
// Outputs magnitude for each range bin

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer ShiftedRef {
    vec2 shifted[]; // Doppler-shifted reference (complex)
};

layout(std430, binding = 1) readonly buffer Surveillance {
    vec2 surv[]; // Surveillance signal (complex)
};

layout(std430, binding = 2) writeonly buffer Output {
    float magnitude[]; // Output magnitude for each range bin
};

layout(push_constant) uniform PushConsts {
    uint n_samples;     // Number of samples in signals
    uint n_range_bins;  // Number of range delay bins
    uint doppler_idx;   // Current Doppler bin index (for output offset)
} pc;

// Shared memory for partial sums reduction
shared vec2 partial_sums[256];

void main() {
    uint range_bin = gl_WorkGroupID.x;

    if (range_bin >= pc.n_range_bins) return;

    uint tid = gl_LocalInvocationID.x;
    uint workgroup_size = gl_WorkGroupSize.x;

    // Each thread computes partial correlation sum
    vec2 local_sum = vec2(0.0, 0.0);

    // Number of valid samples for this range delay
    uint max_samples = pc.n_samples - range_bin;

    // Each thread handles multiple samples with stride
    for (uint i = tid; i < max_samples; i += workgroup_size) {
        vec2 s = shifted[i];
        vec2 v = surv[i + range_bin];

        // shifted * conj(surv)
        // (s.x + j*s.y) * (v.x - j*v.y)
        // = (s.x*v.x + s.y*v.y) + j*(s.y*v.x - s.x*v.y)
        local_sum.x += s.x * v.x + s.y * v.y;
        local_sum.y += s.y * v.x - s.x * v.y;
    }

    partial_sums[tid] = local_sum;
    barrier();

    // Reduction in shared memory
    for (uint s = workgroup_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        barrier();
    }

    // Thread 0 writes the final magnitude
    if (tid == 0) {
        vec2 corr = partial_sums[0];
        float mag = sqrt(corr.x * corr.x + corr.y * corr.y);
        magnitude[pc.doppler_idx * pc.n_range_bins + range_bin] = mag;
    }
}
