#ifndef NANOLLM_ROPE_OPS_H
#define NANOLLM_ROPE_OPS_H

#include <cmath>

namespace nanollm_rope {

// Standard RoFormer base frequency (matches python/model.py).
inline void fill_inv_freq(int head_dim, float* inv_freq_out) {
    const int half = head_dim / 2;
    for (int i = 0; i < half; ++i) {
        inv_freq_out[i] = 1.0f / std::pow(10000.0f, static_cast<float>(i) / static_cast<float>(half));
    }
}

// Apply rotary embedding in-place to one head vector [d_k] at absolute position.
inline void apply_head(float* vec, int d_k, int position, const float* inv_freq) {
    const int half = d_k / 2;
    for (int i = 0; i < half; ++i) {
        const float angle = static_cast<float>(position) * inv_freq[i];
        const float cos_a = std::cos(angle);
        const float sin_a = std::sin(angle);
        const float x1 = vec[2 * i];
        const float x2 = vec[2 * i + 1];
        vec[2 * i] = x1 * cos_a - x2 * sin_a;
        vec[2 * i + 1] = x1 * sin_a + x2 * cos_a;
    }
}

inline void apply_heads(float* vec, int n_heads, int d_k, int position, const float* inv_freq) {
    for (int h = 0; h < n_heads; ++h) {
        apply_head(vec + h * d_k, d_k, position, inv_freq);
    }
}

}  // namespace nanollm_rope

#endif  // NANOLLM_ROPE_OPS_H
