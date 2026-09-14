#pragma once

#include "../curve/bandersnatch.cuh"

namespace cuda_verkle {

// Bandersnatch curve parameters (Twisted Edwards form: a*x^2 + y^2 = 1 + d*x^2*y^2)
// The target is the BLS12-381 scalar field as the base field.

// Compatibility helpers for consumers that need raw Montgomery limbs.  The
// canonical constants themselves live in curve/bandersnatch.cuh; copying them
// here avoids maintaining a second source of truth.
__device__ __host__ inline void get_coeff_a(uint32_t out[8]) {
    const Fp coefficient = curve_coeff_a();
    for (int i = 0; i < 8; ++i) out[i] = coefficient.limbs[i];
}

__device__ __host__ inline void get_coeff_d(uint32_t out[8]) {
    const Fp coefficient = curve_coeff_d();
    for (int i = 0; i < 8; ++i) out[i] = coefficient.limbs[i];
}

} // namespace cuda_verkle
