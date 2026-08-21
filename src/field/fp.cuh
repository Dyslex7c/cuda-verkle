#pragma once
#include <cstdint>

// when compiling with a non-CUDA compiler (e.g. g++, clang++), __device__ and __host__ are not defined. 
// We define them as empty macros so the same code compiles for host-only testing.
#ifndef __CUDACC__
  #ifndef __device__
    #define __device__
  #endif
  #ifndef __host__
    #define __host__
  #endif
#endif

// BLS12-381 scalar field modulus p = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
static constexpr uint32_t FP_MODULUS[8] = {
    0x00000001, 0xffffffff, 0xfffe5bfe, 0x53bda402, 0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753
};

// Montgomery R = 2^256 mod p
static constexpr uint32_t FP_R[8] = {
    0xfffffffe, 0x00000001, 0x00034802, 0x5884b7fa, 0xecbc4ff5, 0x998c4fef, 0xacc5056f, 0x1824b159
};

// Montgomery R^2 = 2^512 mod p
static constexpr uint32_t FP_R2[8] = {
    0xf3f29c6d, 0xc999e990, 0x87925c23, 0x2b6cedcb, 0x7254398f, 0x05d31496, 0x9f59ff11, 0x0748d9d9
};

// Montgomery n' = -p^{-1} mod 2^32
static constexpr uint32_t FP_INV = 0xffffffff;

struct Fp {
    uint32_t limbs[8];
};

#ifdef FP_ZERO
#undef FP_ZERO
#endif

static constexpr Fp FP_ZERO = {{0, 0, 0, 0, 0, 0, 0, 0}};
static constexpr Fp FP_ONE = {{
    0xfffffffe, 0x00000001, 0x00034802, 0x5884b7fa, 
    0xecbc4ff5, 0x998c4fef, 0xacc5056f, 0x1824b159
}}; // R mod p

__device__ __host__ inline bool fp_eq(const Fp& a, const Fp& b) {
    for (int i = 0; i < 8; ++i) {
        if (a.limbs[i] != b.limbs[i]) return false;
    }
    return true;
}

__device__ __host__ inline bool fp_is_zero(const Fp& a) {
    for (int i = 0; i < 8; ++i) {
        if (a.limbs[i] != 0) return false;
    }
    return true;
}

__device__ __host__ inline int fp_cmp(const uint32_t a[8], const uint32_t b[8]) {
    for (int i = 7; i >= 0; --i) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

__device__ __host__ inline Fp fp_add(const Fp& a, const Fp& b) {
    Fp res;
#ifdef __CUDA_ARCH__
    uint32_t carry = 0;
    asm("add.cc.u32 %0, %1, %2;" : "=r"(res.limbs[0]) : "r"(a.limbs[0]), "r"(b.limbs[0]));
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(res.limbs[1]) : "r"(a.limbs[1]), "r"(b.limbs[1]));
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(res.limbs[2]) : "r"(a.limbs[2]), "r"(b.limbs[2]));
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(res.limbs[3]) : "r"(a.limbs[3]), "r"(b.limbs[3]));
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(res.limbs[4]) : "r"(a.limbs[4]), "r"(b.limbs[4]));
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(res.limbs[5]) : "r"(a.limbs[5]), "r"(b.limbs[5]));
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(res.limbs[6]) : "r"(a.limbs[6]), "r"(b.limbs[6]));
    asm("addc.cc.u32 %0, %1, %2;" : "=r"(res.limbs[7]) : "r"(a.limbs[7]), "r"(b.limbs[7]));
#else
    uint64_t carry = 0;
    for (int i = 0; i < 8; ++i) {
        uint64_t sum = (uint64_t)a.limbs[i] + b.limbs[i] + carry;
        res.limbs[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
#endif

    if (fp_cmp(res.limbs, FP_MODULUS) >= 0) {
        Fp sub_res;
#ifdef __CUDA_ARCH__
        asm("sub.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[0]) : "r"(res.limbs[0]), "r"(FP_MODULUS[0]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[1]) : "r"(res.limbs[1]), "r"(FP_MODULUS[1]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[2]) : "r"(res.limbs[2]), "r"(FP_MODULUS[2]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[3]) : "r"(res.limbs[3]), "r"(FP_MODULUS[3]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[4]) : "r"(res.limbs[4]), "r"(FP_MODULUS[4]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[5]) : "r"(res.limbs[5]), "r"(FP_MODULUS[5]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[6]) : "r"(res.limbs[6]), "r"(FP_MODULUS[6]));
        asm("subc.u32 %0, %1, %2;"    : "=r"(sub_res.limbs[7]) : "r"(res.limbs[7]), "r"(FP_MODULUS[7]));
#else
        uint64_t borrow = 0;
        for (int i = 0; i < 8; ++i) {
            uint64_t diff = (uint64_t)res.limbs[i] - FP_MODULUS[i] - borrow;
            sub_res.limbs[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
#endif
        return sub_res;
    }
    return res;
}

__device__ __host__ inline Fp fp_sub(const Fp& a, const Fp& b) {
    Fp res;
#ifdef __CUDA_ARCH__
    asm("sub.cc.u32 %0, %1, %2;" : "=r"(res.limbs[0]) : "r"(a.limbs[0]), "r"(b.limbs[0]));
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(res.limbs[1]) : "r"(a.limbs[1]), "r"(b.limbs[1]));
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(res.limbs[2]) : "r"(a.limbs[2]), "r"(b.limbs[2]));
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(res.limbs[3]) : "r"(a.limbs[3]), "r"(b.limbs[3]));
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(res.limbs[4]) : "r"(a.limbs[4]), "r"(b.limbs[4]));
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(res.limbs[5]) : "r"(a.limbs[5]), "r"(b.limbs[5]));
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(res.limbs[6]) : "r"(a.limbs[6]), "r"(b.limbs[6]));
    uint32_t borrow;
    asm("subc.cc.u32 %0, %1, %2;" : "=r"(res.limbs[7]) : "r"(a.limbs[7]), "r"(b.limbs[7]));
    asm("subc.u32 %0, 0, 0;" : "=r"(borrow)); // capture carry flag
#else
    uint64_t borrow = 0;
    for (int i = 0; i < 8; ++i) {
        uint64_t diff = (uint64_t)a.limbs[i] - b.limbs[i] - borrow;
        res.limbs[i] = (uint32_t)diff;
        borrow = (diff >> 63) & 1;
    }
#endif

#ifdef __CUDA_ARCH__
    if (borrow) {
#else
    if (borrow) {
#endif
        Fp add_res;
#ifdef __CUDA_ARCH__
        asm("add.cc.u32 %0, %1, %2;" : "=r"(add_res.limbs[0]) : "r"(res.limbs[0]), "r"(FP_MODULUS[0]));
        asm("addc.cc.u32 %0, %1, %2;" : "=r"(add_res.limbs[1]) : "r"(res.limbs[1]), "r"(FP_MODULUS[1]));
        asm("addc.cc.u32 %0, %1, %2;" : "=r"(add_res.limbs[2]) : "r"(res.limbs[2]), "r"(FP_MODULUS[2]));
        asm("addc.cc.u32 %0, %1, %2;" : "=r"(add_res.limbs[3]) : "r"(res.limbs[3]), "r"(FP_MODULUS[3]));
        asm("addc.cc.u32 %0, %1, %2;" : "=r"(add_res.limbs[4]) : "r"(res.limbs[4]), "r"(FP_MODULUS[4]));
        asm("addc.cc.u32 %0, %1, %2;" : "=r"(add_res.limbs[5]) : "r"(res.limbs[5]), "r"(FP_MODULUS[5]));
        asm("addc.cc.u32 %0, %1, %2;" : "=r"(add_res.limbs[6]) : "r"(res.limbs[6]), "r"(FP_MODULUS[6]));
        asm("addc.u32 %0, %1, %2;" : "=r"(add_res.limbs[7]) : "r"(res.limbs[7]), "r"(FP_MODULUS[7]));
#else
        uint64_t carry = 0;
        for (int i = 0; i < 8; ++i) {
            uint64_t sum = (uint64_t)res.limbs[i] + FP_MODULUS[i] + carry;
            add_res.limbs[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
#endif
        return add_res;
    }
    return res;
}

__device__ __host__ inline Fp fp_neg(const Fp& a) {
    if (fp_is_zero(a)) return FP_ZERO;
    return fp_sub(FP_ZERO, a);
}

__device__ __host__ inline Fp fp_mul(const Fp& a, const Fp& b) {
    uint32_t t[9] = {0};

    for (int i = 0; i < 8; ++i) {
        uint32_t ai = a.limbs[i];
#ifdef __CUDA_ARCH__
        // t += a[i] * b
        uint32_t carry1 = 0;
        for (int j = 0; j < 8; ++j) {
            uint32_t lo, hi;
            asm("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(lo) : "r"(ai), "r"(b.limbs[j]), "r"(t[j]));
            asm("madc.hi.cc.u32 %0, %1, %2, 0;" : "=r"(hi) : "r"(ai), "r"(b.limbs[j]));
            
            asm("add.cc.u32 %0, %1, %2;" : "=r"(t[j]) : "r"(lo), "r"(carry1));
            asm("addc.u32 %0, %1, 0;" : "=r"(carry1) : "r"(hi));
        }
        t[8] = carry1;

        // m = t[0] * n'
        uint32_t m = t[0] * FP_INV;

        // t += m * p
        uint32_t carry2 = 0;
        for (int j = 0; j < 8; ++j) {
            uint32_t lo, hi;
            asm("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(lo) : "r"(m), "r"(FP_MODULUS[j]), "r"(t[j]));
            asm("madc.hi.cc.u32 %0, %1, %2, 0;" : "=r"(hi) : "r"(m), "r"(FP_MODULUS[j]));
            
            asm("add.cc.u32 %0, %1, %2;" : "=r"(t[j]) : "r"(lo), "r"(carry2));
            asm("addc.u32 %0, %1, 0;" : "=r"(carry2) : "r"(hi));
        }
        asm("add.cc.u32 %0, %1, %2;" : "=r"(t[8]) : "r"(t[8]), "r"(carry2));
        // we know that carry out of t[8] cannot happen because t[8] <= 1
        
        // shift right
        for (int j = 0; j < 8; ++j) {
            t[j] = t[j+1];
        }
        t[8] = 0;
#else
        uint64_t carry1 = 0;
        for (int j = 0; j < 8; ++j) {
            uint64_t sum = (uint64_t)t[j] + (uint64_t)ai * b.limbs[j] + carry1;
            t[j] = (uint32_t)sum;
            carry1 = sum >> 32;
        }
        t[8] = (uint32_t)carry1;

        uint32_t m = t[0] * FP_INV;

        uint64_t carry2 = 0;
        for (int j = 0; j < 8; ++j) {
            uint64_t sum = (uint64_t)t[j] + (uint64_t)m * FP_MODULUS[j] + carry2;
            t[j] = (uint32_t)sum;
            carry2 = sum >> 32;
        }
        t[8] += (uint32_t)carry2;

        for (int j = 0; j < 8; ++j) {
            t[j] = t[j+1];
        }
        t[8] = 0;
#endif
    }

    Fp res;
    for (int j = 0; j < 8; ++j) res.limbs[j] = t[j];
    
    if (fp_cmp(res.limbs, FP_MODULUS) >= 0) {
        Fp sub_res;
#ifdef __CUDA_ARCH__
        asm("sub.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[0]) : "r"(res.limbs[0]), "r"(FP_MODULUS[0]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[1]) : "r"(res.limbs[1]), "r"(FP_MODULUS[1]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[2]) : "r"(res.limbs[2]), "r"(FP_MODULUS[2]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[3]) : "r"(res.limbs[3]), "r"(FP_MODULUS[3]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[4]) : "r"(res.limbs[4]), "r"(FP_MODULUS[4]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[5]) : "r"(res.limbs[5]), "r"(FP_MODULUS[5]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[6]) : "r"(res.limbs[6]), "r"(FP_MODULUS[6]));
        asm("subc.u32 %0, %1, %2;"    : "=r"(sub_res.limbs[7]) : "r"(res.limbs[7]), "r"(FP_MODULUS[7]));
#else
        uint64_t borrow = 0;
        for (int i = 0; i < 8; ++i) {
            uint64_t diff = (uint64_t)res.limbs[i] - FP_MODULUS[i] - borrow;
            sub_res.limbs[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
#endif
        return sub_res;
    }

    return res;
}

__device__ __host__ inline Fp fp_sqr(const Fp& a) {
    return fp_mul(a, a);
}

// Convert from raw limbs to Montgomery form (a * R mod p)
__device__ __host__ inline Fp fp_from_raw(const uint32_t limbs[8]) {
    Fp a;
    for (int i = 0; i < 8; ++i) a.limbs[i] = limbs[i];
    Fp r2;
    for (int i = 0; i < 8; ++i) r2.limbs[i] = FP_R2[i];
    return fp_mul(a, r2);
}

// Convert from Montgomery form to raw limbs (a * 1 mod p)
__device__ __host__ inline void fp_to_raw(const Fp& a, uint32_t limbs[8]) {
    Fp one = FP_ZERO;
    one.limbs[0] = 1;
    Fp res = fp_mul(a, one);
    for (int i = 0; i < 8; ++i) limbs[i] = res.limbs[i];
}

__device__ __host__ inline Fp fp_from_u64(uint64_t val) {
    uint32_t raw[8] = { (uint32_t)val, (uint32_t)(val >> 32), 0, 0, 0, 0, 0, 0 };
    return fp_from_raw(raw);
}

__device__ __host__ inline Fp fp_pow(Fp base, const uint32_t exp[8]) {
    Fp res = FP_ONE;
    for (int i = 7; i >= 0; --i) {
        for (int j = 31; j >= 0; --j) {
            res = fp_sqr(res);
            if ((exp[i] >> j) & 1) {
                res = fp_mul(res, base);
            }
        }
    }
    return res;
}

__device__ __host__ inline Fp fp_inv(const Fp& a) {
    // Fermat's Little Theorem: a^(p-2) mod p (Finally something learnt from college (Discrete Mathematics) is applicable here)
    // p-2 = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001 - 2 = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfefffffffeffffffff
    uint32_t p_minus_2[8] = {
        0xffffffff, 0xfffffffe, 0xfffe5bfe, 0x53bda402, 
        0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753
    };
    return fp_pow(a, p_minus_2);
}
