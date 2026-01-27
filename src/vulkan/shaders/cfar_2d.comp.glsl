#version 450
// 2D Cell-Averaging CFAR Detector for Range-Doppler Maps
// Computes threshold based on average of reference cells
// Outputs binary detection mask

layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer Input {
    float power[]; // Input power/magnitude [n_doppler x n_range]
};

layout(std430, binding = 1) writeonly buffer Output {
    uint detections[]; // Binary output (0 or 1)
};

layout(push_constant) uniform PushConsts {
    uint n_doppler;
    uint n_range;
    uint guard_range;
    uint guard_doppler;
    uint ref_range;
    uint ref_doppler;
    float pfa_factor;
} pc;

void main() {
    uint doppler = gl_GlobalInvocationID.x;
    uint range = gl_GlobalInvocationID.y;

    if (doppler >= pc.n_doppler || range >= pc.n_range) return;

    uint cell_idx = doppler * pc.n_range + range;
    float cell_value = power[cell_idx];

    // Compute average of reference cells
    float sum = 0.0;
    uint count = 0;

    int total_guard_d = int(pc.guard_doppler);
    int total_guard_r = int(pc.guard_range);
    int total_ref_d = int(pc.guard_doppler + pc.ref_doppler);
    int total_ref_r = int(pc.guard_range + pc.ref_range);

    for (int dd = -total_ref_d; dd <= total_ref_d; ++dd) {
        for (int dr = -total_ref_r; dr <= total_ref_r; ++dr) {
            // Skip cell under test
            if (dd == 0 && dr == 0) continue;

            // Skip guard cells
            if (abs(dd) <= total_guard_d && abs(dr) <= total_guard_r) continue;

            int nd = int(doppler) + dd;
            int nr = int(range) + dr;

            // Bounds check
            if (nd >= 0 && nd < int(pc.n_doppler) && nr >= 0 && nr < int(pc.n_range)) {
                sum += power[nd * pc.n_range + nr];
                count++;
            }
        }
    }

    // Compute threshold
    float avg = (count > 0) ? (sum / float(count)) : 0.0;
    float threshold = pc.pfa_factor * avg;

    // Detection decision
    detections[cell_idx] = (cell_value > threshold) ? 1u : 0u;
}
