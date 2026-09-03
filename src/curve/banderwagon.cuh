#pragma once

#include "bandersnatch.cuh"
#include "../field/fp.cuh"
#include "../field/fr.cuh"
#include <stdint.h>

// Banderwagon is a quotient group of Bandersnatch by its order-2 point.
// Two Bandersnatch points (x1,y1) and (x2,y2) are equal in Banderwagon iff x1*y2 == x2*y1. 
// This eliminates the ambiguity from the cofactor-4 structure of Bandersnatch and gives us a clean prime-order group.
struct BanderwagonElement {
    PointExtended point;
};

// Check if two elements are equal in the Banderwagon quotient group.
// In projective coordinates the equality (X1/Z1)*(Y2/Z2) == (X2/Z2)*(Y1/Z1) simplifies to X1*Y2 == X2*Y1 (Z terms cancel).
__device__ __host__ inline bool bw_eq(const BanderwagonElement& a, const BanderwagonElement& b) {
    Fp lhs = fp_mul(a.point.X, b.point.Y);
    Fp rhs = fp_mul(b.point.X, a.point.Y);
    return fp_eq(lhs, rhs);
}

// Returns the identity element of the Banderwagon group.
__device__ __host__ inline BanderwagonElement bw_identity() {
    return {point_identity()};
}

// Adds two elements in the Banderwagon group.
__device__ __host__ inline BanderwagonElement bw_add(const BanderwagonElement& a, const BanderwagonElement& b) {
    return {point_add(a.point, b.point)};
}

// Doubles an element in the Banderwagon group.
__device__ __host__ inline BanderwagonElement bw_double(const BanderwagonElement& a) {
    return {point_double(a.point)};
}

// Scalar multiplication of an element in the Banderwagon group.
__device__ __host__ inline BanderwagonElement bw_scalar_mul(const BanderwagonElement& a, const Fr& scalar) {
    return {scalar_mul(a.point, scalar)};
}

// negates an element in the Banderwagon group.
// in twisted Edwards curves, negation flips the x-coordinate: (x,y) -> (-x,y)
__device__ __host__ inline BanderwagonElement bw_neg(const BanderwagonElement& a) {
    return {point_neg(a.point)};
}

// Check the Banderwagon/Rust reference sign convention for serialization.
// It defines positive as y > -y, which is exactly y > (p - 1) / 2 for a
// canonical field element. This selects the same x or -x representative as
// Element::to_bytes() in rust-verkle.
__device__ __host__ inline bool fp_is_positive(const Fp& y) {
    // (p-1)/2 in little-endian 32-bit limbs:
    // p = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
    // (p-1)/2 = 0x39f6d3a994cebea4199cec0404d0ec02a9ded2017fff2dff7fffffff80000000
    static constexpr uint32_t HALF_P[8] = {
        0x80000000, 0x7fffffff, 0x7fff2dff, 0xa9ded201,
        0x04d0ec02, 0x199cec04, 0x94cebea4, 0x39f6d3a9
    };

    uint32_t raw[8];
    fp_to_raw(y, raw);

    // Compare raw against HALF_P (big-endian comparison).
    for (int i = 7; i >= 0; --i) {
        if (raw[i] > HALF_P[i]) return true;
        if (raw[i] < HALF_P[i]) return false;
    }
    return false; // equality would mean y == -y, which is not positive
}

// serializes a Banderwagon element into a 32-byte array.
__device__ __host__ inline void bw_to_bytes(const BanderwagonElement& e, uint8_t out[32]) {
    PointAffine affine = point_to_affine(e.point);
    Fp x = affine.x;

    if (!fp_is_positive(affine.y)) {
        x = fp_neg(x);
    }

    uint32_t raw[8];
    fp_to_raw(x, raw);

    for (int i = 0; i < 8; ++i) {
        uint32_t limb = raw[7 - i]; // MSB limb first
        out[i * 4 + 0] = (limb >> 24) & 0xFF;
        out[i * 4 + 1] = (limb >> 16) & 0xFF;
        out[i * 4 + 2] = (limb >>  8) & 0xFF;
        out[i * 4 + 3] = limb & 0xFF;
    }
}

// verify that (1 - a*x^2) is a quadratic residue in Fp.
// This ensures the point is in the correct prime-order subgroup of Banderwagon.
// uses the Legendre symbol: val^((p-1)/2) must equal 1.
__device__ __host__ inline bool bw_subgroup_check(const BanderwagonElement& e) {
    PointAffine affine = point_to_affine(e.point);
    Fp x_sqr = fp_sqr(affine.x);

    Fp a_x_sqr = fp_mul(COEFF_A, x_sqr);

    Fp one = FP_ONE;
    Fp val = fp_sub(one, a_x_sqr);

    // check quadratic residue: val^((p-1)/2) == 1
    // (p-1)/2 in little-endian 32-bit limbs:
    uint32_t exp[8] = {
        0x80000000, 0x7fffffff, 0x7fff2dff, 0xa9ded201,
        0x04d0ec02, 0x199cec04, 0x94cebea4, 0x39f6d3a9
    };

    Fp qr_check = fp_pow(val, exp);
    return fp_eq(qr_check, one);
}
