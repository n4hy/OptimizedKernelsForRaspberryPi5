#version 450
// Cross-Ambiguity Function - Phase 1: Doppler Shift
// Applies Doppler frequency shift to reference signal
// shifted = ref * exp(j * 2 * pi * fd * t)

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer RefSignal {
    vec2 ref[]; // Complex: (real, imag)
};

layout(std430, binding = 1) writeonly buffer ShiftedSignal {
    vec2 shifted[]; // Complex output
};

layout(push_constant) uniform PushConsts {
    uint n_samples;
    float phase_step; // 2 * pi * doppler_freq / sample_rate
} pc;

const float PI = 3.14159265358979323846;

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if (idx >= pc.n_samples) return;

    // Calculate phase for this sample
    float phase = pc.phase_step * float(idx);

    // exp(j * phase) = cos(phase) + j * sin(phase)
    float cos_p = cos(phase);
    float sin_p = sin(phase);

    vec2 r = ref[idx];

    // Complex multiply: ref * exp(j*phase)
    // (r.x + j*r.y) * (cos_p + j*sin_p)
    // = (r.x*cos_p - r.y*sin_p) + j*(r.x*sin_p + r.y*cos_p)
    shifted[idx] = vec2(
        r.x * cos_p - r.y * sin_p,
        r.x * sin_p + r.y * cos_p
    );
}
