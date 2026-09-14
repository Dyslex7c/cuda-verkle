#pragma once

#include "../field/fp.cuh"
#include "../field/fr.cuh"

// COEFF_A = -5 mod p, in Montgomery form  
static constexpr uint32_t COEFF_A_MONT_LIMBS[8] = {
    0x0000000c, 0xfffffff4, 0xffec4ff3, 0xece3b023, 0x7396203f, 0x66b62060, 0xf361df62, 0x6f23d7e5
};

// COEFF_D in Montgomery form
static constexpr uint32_t COEFF_D_MONT_LIMBS[8] = {
    0x47a2c730, 0xa8dced1b, 0xad3cccc7, 0x381c065a, 0x188351f8, 0x53ff52e1, 0x990fe940, 0x362e8d63
};

// field element for curve parameter a
static constexpr Fp COEFF_A = {
    COEFF_A_MONT_LIMBS[0], COEFF_A_MONT_LIMBS[1], COEFF_A_MONT_LIMBS[2], COEFF_A_MONT_LIMBS[3],
    COEFF_A_MONT_LIMBS[4], COEFF_A_MONT_LIMBS[5], COEFF_A_MONT_LIMBS[6], COEFF_A_MONT_LIMBS[7]
};

// field element for curve parameter d
static constexpr Fp COEFF_D = {
    COEFF_D_MONT_LIMBS[0], COEFF_D_MONT_LIMBS[1], COEFF_D_MONT_LIMBS[2], COEFF_D_MONT_LIMBS[3],
    COEFF_D_MONT_LIMBS[4], COEFF_D_MONT_LIMBS[5], COEFF_D_MONT_LIMBS[6], COEFF_D_MONT_LIMBS[7]
};

// point on twisted Edwards curve in extended projective coordinates (X:Y:T:Z)
// Represents the affine point (x, y) where x = X/Z, y = Y/Z, and T = XY/Z
struct PointExtended {
    Fp X, Y, T, Z;
};

// point on twisted Edwards curve in affine coordinates (x, y)
struct PointAffine {
    Fp x, y;
};

__device__ __host__ inline PointExtended point_identity() {
    return {FP_MONT_ZERO, FP_MONT_ONE, FP_MONT_ZERO, FP_MONT_ONE};
}

// Adds two points in extended projective coordinates
__device__ __host__ inline PointExtended point_add(const PointExtended& P, const PointExtended& Q) {
    Fp A = fp_mul(P.X, Q.X);

    Fp B = fp_mul(P.Y, Q.Y);
    
    Fp C = fp_mul(fp_mul(P.T, COEFF_D), Q.T);

    Fp D = fp_mul(P.Z, Q.Z);

    Fp P_X_plus_Y = fp_add(P.X, P.Y);

    Fp Q_X_plus_Y = fp_add(Q.X, Q.Y);

    Fp E = fp_sub(fp_sub(fp_mul(P_X_plus_Y, Q_X_plus_Y), A), B);

    Fp F = fp_sub(D, C);

    Fp G = fp_add(D, C);

    Fp H = fp_sub(B, fp_mul(COEFF_A, A));


    Fp X3 = fp_mul(E, F);

    Fp Y3 = fp_mul(G, H);

    Fp T3 = fp_mul(E, H);

    Fp Z3 = fp_mul(F, G);

    return {X3, Y3, T3, Z3};
}

// doubles a point in extended projective coordinates
__device__ __host__ inline PointExtended point_double(const PointExtended& P) {

    Fp A = fp_sqr(P.X);

    Fp B = fp_sqr(P.Y);

    Fp Z_sqr = fp_sqr(P.Z);
    Fp C = fp_add(Z_sqr, Z_sqr);

    Fp D = fp_mul(COEFF_A, A);

    Fp X_plus_Y = fp_add(P.X, P.Y);
    Fp E = fp_sub(fp_sub(fp_sqr(X_plus_Y), A), B);

    Fp G = fp_add(D, B);

    Fp F = fp_sub(G, C);

    Fp H = fp_sub(D, B);


    Fp X3 = fp_mul(E, F);

    Fp Y3 = fp_mul(G, H);

    Fp T3 = fp_mul(E, H);

    Fp Z3 = fp_mul(F, G);

    return {X3, Y3, T3, Z3};
}

// scalar multiplication using double-and-add (MSB to LSB)
__device__ __host__ inline PointExtended scalar_mul(const PointExtended& P, const Fr& scalar) {
    uint32_t raw_limbs[8];
    fr_to_raw(scalar, raw_limbs);
    PointExtended result = point_identity();
    bool found_one = false;
    
    // process from MSB to LSB
    for (int i = 7; i >= 0; --i) {
        uint32_t limb = raw_limbs[i];
        for (int j = 31; j >= 0; --j) {
            bool bit = (limb >> j) & 1;
            
            if (found_one) {
                result = point_double(result);
            }
            
            if (bit) {
                result = found_one ? point_add(result, P) : P;
                found_one = true;
            }
        }
    }
    return result;
}

// convert from extended projective coordinates to affine coordinates
__device__ __host__ inline PointAffine point_to_affine(const PointExtended& P) {
    Fp z_inv = fp_inv(P.Z);

    Fp x = fp_mul(P.X, z_inv);

    Fp y = fp_mul(P.Y, z_inv);

    return {x, y};
}

// convert from affine coordinates to extended projective coordinates
__device__ __host__ inline PointExtended point_from_affine(const PointAffine& P) {
    Fp T = fp_mul(P.x, P.y);
    return {P.x, P.y, T, FP_MONT_ONE};
}

// check if point is the identity point
__device__ __host__ inline bool point_is_identity(const PointExtended& P) {
    return fp_is_zero(P.X) && fp_eq(P.Y, P.Z);
}

// negate a point on the twisted Edwards curve
__device__ __host__ inline PointExtended point_neg(const PointExtended& P) {
    return {fp_neg(P.X), P.Y, fp_neg(P.T), P.Z};
}
