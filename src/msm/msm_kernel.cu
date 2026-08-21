#include "msm_kernel.cuh"

// Extract bits [window_idx*w, window_idx*w + w) from the scalar
__host__ __device__ inline uint32_t get_window(const Fr& scalar, int window_idx, int w) {
    uint32_t raw[8];
    fr_to_raw(scalar, raw);
    int bit_start = window_idx * w;
    int limb_idx = bit_start / 32;
    int bit_offset = bit_start % 32;
    uint32_t mask = (1u << w) - 1;
    uint32_t digit = (raw[limb_idx] >> bit_offset) & mask;
    // Handle spanning two limbs
    if (bit_offset + w > 32 && limb_idx + 1 < 8) {
        digit |= (raw[limb_idx + 1] << (32 - bit_offset)) & mask;
    }
    return digit;
}

// for each (scalar, point), compute scalar_mul and accumulate
__host__ __device__ PointExtended msm_naive(const Fr scalars[], const Fp px[], const Fp py[], int n) {
    PointExtended result = point_identity();
    for (int i = 0; i < n; ++i) {
        if (fr_is_zero(scalars[i])) continue;  // skip zero scalars
        PointAffine base = {px[i], py[i]};
        PointExtended base_ext = point_from_affine(base);
        PointExtended term = scalar_mul(base_ext, scalars[i]);
        result = point_add(result, term);
    }
    return result;
}

__host__ __device__ PointExtended msm_cpu_reference(
    const Fr scalars[],
    const Fp point_x[],
    const Fp point_y[],
    int n) {
    return msm_naive(scalars, point_x, point_y, n);
}

__host__ __device__ PointExtended msm_compute(
    const Fr scalars[],
    const Fp point_x[],
    const Fp point_y[],
    int n) {
    // The CRS has a fixed width. This is a public function, so protect callers from indexing past the supplied fixed-size point arrays.
    if (n <= 0) return point_identity();
    if (n > MSM_SIZE) n = MSM_SIZE;
    
    int w = 8;
    int num_windows = (253 + w - 1) / w; // ceil(253 / 8) = 32
    
    PointExtended total = point_identity();
    
    // For w=8, 2^w = 256
    PointExtended buckets[256];

    for (int window_idx = num_windows - 1; window_idx >= 0; --window_idx) {
        // Shift total by w bits
        for (int i = 0; i < w; ++i) {
            total = point_double(total);
        }
        
        // Initialize buckets to identity
        for (int i = 1; i < 256; ++i) {
            buckets[i] = point_identity();
        }
        
        for (int i = 0; i < n; ++i) {
            uint32_t digit = get_window(scalars[i], window_idx, w);
            if (digit != 0) {
                PointAffine base = {point_x[i], point_y[i]};
                PointExtended base_ext = point_from_affine(base);
                buckets[digit] = point_add(buckets[digit], base_ext);
            }
        }
        
        PointExtended running_sum = point_identity();
        PointExtended partial = point_identity();
        
        for (int j = 255; j >= 1; --j) {
            running_sum = point_add(running_sum, buckets[j]);
            partial = point_add(partial, running_sum);
        }
        
        total = point_add(total, partial);
    }
    
    return total;
}
