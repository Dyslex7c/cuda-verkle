#pragma once
#include <cstdint>

// CUDA compatibility: when compiling with a non-CUDA compiler (e.g. g++, clang++),
// __device__ and __host__ are not defined. We define them as empty macros so the
// same code compiles for host-only testing.
#ifndef __CUDACC__
  #ifndef __device__
    #define __device__
  #endif
  #ifndef __host__
    #define __host__
  #endif
#endif

// Bandersnatch subgroup order
// = 0x1cfb69d4ca675f520cce760202687600ff8f87007419047174fd06b52876e7e1
static constexpr uint32_t FR_MODULUS[8] = {
    0x2876e7e1, 0x74fd06b5, 0x74190471, 0xff8f8700, 0x02687600, 0x0cce7602, 0xca675f52, 0x1cfb69d4
};

// Montgomery R = 2^256 mod n
static constexpr uint32_t FR_R[8] = {
    0xbc48c0f8, 0x5817ca56, 0x5f37dc74, 0x0383c7fc, 0xecbc4ff8, 0x998c4fef, 0xacc5056f, 0x1824b159
};

// Montgomery R^2 = 2^512 mod n
static constexpr uint32_t FR_R2[8] = {
    0x58db47cb, 0xdbb4f5d6, 0x7fecb938, 0x40fa7ca2, 0xc0055cea, 0xaa9e6dae, 0xb14aec7d, 0x0ae793dd
};

// Montgomery n' = -n^{-1} mod 2^32
static constexpr uint32_t FR_INV = 0x5cc063df;

struct Fr {
    uint32_t limbs[8];
};

static constexpr Fr FR_ZERO = {{0, 0, 0, 0, 0, 0, 0, 0}};
static constexpr Fr FR_ONE = {{
    0xbc48c0f8, 0x5817ca56, 0x5f37dc74, 0x0383c7fc, 
    0xecbc4ff8, 0x998c4fef, 0xacc5056f, 0x1824b159
}}; // R mod n

// --- Fr Function Declarations ---

__device__ __host__ inline bool fr_eq(const Fr& a, const Fr& b) {
    for (int i = 0; i < 8; ++i) {
        if (a.limbs[i] != b.limbs[i]) return false;
    }
    return true;
}

__device__ __host__ inline bool fr_is_zero(const Fr& a) {
    for (int i = 0; i < 8; ++i) {
        if (a.limbs[i] != 0) return false;
    }
    return true;
}

__device__ __host__ inline int fr_cmp(const uint32_t a[8], const uint32_t b[8]) {
    for (int i = 7; i >= 0; --i) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

__device__ __host__ inline Fr fr_add(const Fr& a, const Fr& b) {
    Fr res;
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

    if (fr_cmp(res.limbs, FR_MODULUS) >= 0) {
        Fr sub_res;
#ifdef __CUDA_ARCH__
        asm("sub.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[0]) : "r"(res.limbs[0]), "r"(FR_MODULUS[0]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[1]) : "r"(res.limbs[1]), "r"(FR_MODULUS[1]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[2]) : "r"(res.limbs[2]), "r"(FR_MODULUS[2]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[3]) : "r"(res.limbs[3]), "r"(FR_MODULUS[3]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[4]) : "r"(res.limbs[4]), "r"(FR_MODULUS[4]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[5]) : "r"(res.limbs[5]), "r"(FR_MODULUS[5]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[6]) : "r"(res.limbs[6]), "r"(FR_MODULUS[6]));
        asm("subc.u32 %0, %1, %2;"    : "=r"(sub_res.limbs[7]) : "r"(res.limbs[7]), "r"(FR_MODULUS[7]));
#else
        uint64_t borrow = 0;
        for (int i = 0; i < 8; ++i) {
            uint64_t diff = (uint64_t)res.limbs[i] - FR_MODULUS[i] - borrow;
            sub_res.limbs[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
#endif
        return sub_res;
    }
    return res;
}

__device__ __host__ inline Fr fr_sub(const Fr& a, const Fr& b) {
    Fr res;
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
    asm("subc.u32 %0, 0, 0;" : "=r"(borrow)); // Capture carry flag
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
        Fr add_res;
#ifdef __CUDA_ARCH__
        asm("add.cc.u32 %0, %1, %2;" : "=r"(add_res.limbs[0]) : "r"(res.limbs[0]), "r"(FR_MODULUS[0]));
        asm("addc.cc.u32 %0, %1, %2;" : "=r"(add_res.limbs[1]) : "r"(res.limbs[1]), "r"(FR_MODULUS[1]));
        asm("addc.cc.u32 %0, %1, %2;" : "=r"(add_res.limbs[2]) : "r"(res.limbs[2]), "r"(FR_MODULUS[2]));
        asm("addc.cc.u32 %0, %1, %2;" : "=r"(add_res.limbs[3]) : "r"(res.limbs[3]), "r"(FR_MODULUS[3]));
        asm("addc.cc.u32 %0, %1, %2;" : "=r"(add_res.limbs[4]) : "r"(res.limbs[4]), "r"(FR_MODULUS[4]));
        asm("addc.cc.u32 %0, %1, %2;" : "=r"(add_res.limbs[5]) : "r"(res.limbs[5]), "r"(FR_MODULUS[5]));
        asm("addc.cc.u32 %0, %1, %2;" : "=r"(add_res.limbs[6]) : "r"(res.limbs[6]), "r"(FR_MODULUS[6]));
        asm("addc.u32 %0, %1, %2;" : "=r"(add_res.limbs[7]) : "r"(res.limbs[7]), "r"(FR_MODULUS[7]));
#else
        uint64_t carry = 0;
        for (int i = 0; i < 8; ++i) {
            uint64_t sum = (uint64_t)res.limbs[i] + FR_MODULUS[i] + carry;
            add_res.limbs[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
#endif
        return add_res;
    }
    return res;
}

__device__ __host__ inline Fr fr_neg(const Fr& a) {
    if (fr_is_zero(a)) return FR_ZERO;
    return fr_sub(FR_ZERO, a);
}

__device__ __host__ inline Fr fr_mul(const Fr& a, const Fr& b) {
    uint32_t t[9] = {0};

    for (int i = 0; i < 8; ++i) {
        uint32_t ai = a.limbs[i];
#ifdef __CUDA_ARCH__
        // 1. t += a[i] * b
        uint32_t carry1 = 0;
        for (int j = 0; j < 8; ++j) {
            uint32_t lo, hi;
            asm("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(lo) : "r"(ai), "r"(b.limbs[j]), "r"(t[j]));
            asm("madc.hi.cc.u32 %0, %1, %2, 0;" : "=r"(hi) : "r"(ai), "r"(b.limbs[j]));
            
            asm("add.cc.u32 %0, %1, %2;" : "=r"(t[j]) : "r"(lo), "r"(carry1));
            asm("addc.u32 %0, %1, 0;" : "=r"(carry1) : "r"(hi));
        }
        t[8] = carry1;

        // 2. m = t[0] * n'
        uint32_t m = t[0] * FR_INV;

        // 3. t += m * n
        uint32_t carry2 = 0;
        for (int j = 0; j < 8; ++j) {
            uint32_t lo, hi;
            asm("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(lo) : "r"(m), "r"(FR_MODULUS[j]), "r"(t[j]));
            asm("madc.hi.cc.u32 %0, %1, %2, 0;" : "=r"(hi) : "r"(m), "r"(FR_MODULUS[j]));
            
            asm("add.cc.u32 %0, %1, %2;" : "=r"(t[j]) : "r"(lo), "r"(carry2));
            asm("addc.u32 %0, %1, 0;" : "=r"(carry2) : "r"(hi));
        }
        asm("add.cc.u32 %0, %1, %2;" : "=r"(t[8]) : "r"(t[8]), "r"(carry2));
        
        // 4. shift right
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

        uint32_t m = t[0] * FR_INV;

        uint64_t carry2 = 0;
        for (int j = 0; j < 8; ++j) {
            uint64_t sum = (uint64_t)t[j] + (uint64_t)m * FR_MODULUS[j] + carry2;
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

    Fr res;
    for (int j = 0; j < 8; ++j) res.limbs[j] = t[j];
    
    if (fr_cmp(res.limbs, FR_MODULUS) >= 0) {
        Fr sub_res;
#ifdef __CUDA_ARCH__
        asm("sub.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[0]) : "r"(res.limbs[0]), "r"(FR_MODULUS[0]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[1]) : "r"(res.limbs[1]), "r"(FR_MODULUS[1]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[2]) : "r"(res.limbs[2]), "r"(FR_MODULUS[2]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[3]) : "r"(res.limbs[3]), "r"(FR_MODULUS[3]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[4]) : "r"(res.limbs[4]), "r"(FR_MODULUS[4]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[5]) : "r"(res.limbs[5]), "r"(FR_MODULUS[5]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub_res.limbs[6]) : "r"(res.limbs[6]), "r"(FR_MODULUS[6]));
        asm("subc.u32 %0, %1, %2;"    : "=r"(sub_res.limbs[7]) : "r"(res.limbs[7]), "r"(FR_MODULUS[7]));
#else
        uint64_t borrow = 0;
        for (int i = 0; i < 8; ++i) {
            uint64_t diff = (uint64_t)res.limbs[i] - FR_MODULUS[i] - borrow;
            sub_res.limbs[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
#endif
        return sub_res;
    }

    return res;
}

__device__ __host__ inline Fr fr_sqr(const Fr& a) {
    return fr_mul(a, a);
}

// Convert from raw limbs to Montgomery form (a * R mod n)
__device__ __host__ inline Fr fr_from_raw(const uint32_t limbs[8]) {
    Fr a;
    for (int i = 0; i < 8; ++i) a.limbs[i] = limbs[i];
    Fr r2;
    for (int i = 0; i < 8; ++i) r2.limbs[i] = FR_R2[i];
    return fr_mul(a, r2);
}

// Convert from Montgomery form to raw limbs (a * 1 mod n)
__device__ __host__ inline void fr_to_raw(const Fr& a, uint32_t limbs[8]) {
    Fr one = FR_ZERO;
    one.limbs[0] = 1;
    Fr res = fr_mul(a, one);
    for (int i = 0; i < 8; ++i) limbs[i] = res.limbs[i];
}

__device__ __host__ inline Fr fr_from_u64(uint64_t val) {
    uint32_t raw[8] = { (uint32_t)val, (uint32_t)(val >> 32), 0, 0, 0, 0, 0, 0 };
    return fr_from_raw(raw);
}

__device__ __host__ inline Fr fr_pow(Fr base, const uint32_t exp[8]) {
    Fr res = FR_ONE;
    for (int i = 7; i >= 0; --i) {
        for (int j = 31; j >= 0; --j) {
            res = fr_sqr(res);
            if ((exp[i] >> j) & 1) {
                res = fr_mul(res, base);
            }
        }
    }
    return res;
}

__device__ __host__ inline Fr fr_inv(const Fr& a) {
    // Fermat's Little Theorem: a^(n-2) mod n (once again)
    // n-2 = 0x1cfb69d4ca675f520cce760202687600ff8f87007419047174fd06b52876e7e1 - 2
    //     = 0x1cfb69d4ca675f520cce760202687600ff8f87007419047174fd06b52876e7df
    uint32_t n_minus_2[8] = {
        0x2876e7df, 0x74fd06b5, 0x74190471, 0xff8f8700, 
        0x02687600, 0x0cce7602, 0xca675f52, 0x1cfb69d4
    };
    return fr_pow(a, n_minus_2);
}
