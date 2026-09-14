#include <cstdio>
#include <cstdint>
#include <cstring>
#include "../src/util/test_vectors.cuh"
#include "../src/field/fp.cuh"
#include "../src/field/fr.cuh"

static int tests_passed = 0;
static int tests_failed = 0;

static void print_fp_raw(const char* label, const Fp& a) {
    uint32_t raw[8];
    fp_to_raw(a, raw);
    printf("  %s = 0x", label);
    for (int i = 7; i >= 0; --i) printf("%08x", raw[i]);
    printf("\n");
}

#define ASSERT_FP_EQ(a, b, msg) do { \
    if (fp_eq(a, b)) { tests_passed++; printf("  PASS: %s\n", msg); } \
    else { tests_failed++; printf("  FAIL: %s\n", msg); print_fp_raw("got", a); print_fp_raw("exp", b); } \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    if (cond) { tests_passed++; printf("  PASS: %s\n", msg); } \
    else { tests_failed++; printf("  FAIL: %s\n", msg); } \
} while(0)

void test_fp_zero_one() {
    // 0 + 0 = 0
    Fp z = fp_add(fp_mont_zero(), fp_mont_zero());
    ASSERT_FP_EQ(z, fp_mont_zero(), "0 + 0 = 0");

    // 1 * 1 = 1
    Fp one_sq = fp_mul(fp_mont_one(), fp_mont_one());
    ASSERT_FP_EQ(one_sq, fp_mont_one(), "1 * 1 = 1");

    // 0 * 1 = 0
    Fp z_mul = fp_mul(fp_mont_zero(), fp_mont_one());
    ASSERT_FP_EQ(z_mul, fp_mont_zero(), "0 * 1 = 0");

    // 1 + 0 = 1
    Fp one_add = fp_add(fp_mont_one(), fp_mont_zero());
    ASSERT_FP_EQ(one_add, fp_mont_one(), "1 + 0 = 1");

    // 1 - 1 = 0
    Fp one_sub = fp_sub(fp_mont_one(), fp_mont_one());
    ASSERT_FP_EQ(one_sub, fp_mont_zero(), "1 - 1 = 0");

    // -0 = 0
    Fp neg_z = fp_neg(fp_mont_zero());
    ASSERT_FP_EQ(neg_z, fp_mont_zero(), "-0 = 0");
}

void test_fp_add_sub() {
    Fp a = fp_from_u64(42);
    Fp b = fp_from_u64(58);
    Fp c = fp_from_u64(100);

    // 42 + 58 = 100
    Fp sum = fp_add(a, b);
    ASSERT_FP_EQ(sum, c, "42 + 58 = 100");

    // 100 - 58 = 42
    Fp diff = fp_sub(c, b);
    ASSERT_FP_EQ(diff, a, "100 - 58 = 42");

    // a + (-a) = 0
    Fp neg_a = fp_neg(a);
    Fp zero = fp_add(a, neg_a);
    ASSERT_FP_EQ(zero, fp_mont_zero(), "a + (-a) = 0");

    // Commutativity: a + b = b + a
    Fp sum2 = fp_add(b, a);
    ASSERT_FP_EQ(sum, sum2, "a + b = b + a");
}

void test_fp_mul() {
    Fp a = fp_from_u64(7);
    Fp b = fp_from_u64(13);
    Fp c = fp_from_u64(91);

    // 7 * 13 = 91
    Fp prod = fp_mul(a, b);
    ASSERT_FP_EQ(prod, c, "7 * 13 = 91");

    // Commutativity: a * b = b * a
    Fp prod2 = fp_mul(b, a);
    ASSERT_FP_EQ(prod, prod2, "a * b = b * a");

    // Associativity: (a * b) * c = a * (b * c)
    Fp d = fp_from_u64(5);
    Fp ab = fp_mul(a, b);
    Fp ab_d = fp_mul(ab, d);
    Fp bd = fp_mul(b, d);
    Fp a_bd = fp_mul(a, bd);
    ASSERT_FP_EQ(ab_d, a_bd, "(a*b)*c = a*(b*c)");

    // Distributivity: a * (b + c) = a*b + a*c
    Fp b_plus_c = fp_add(b, c);
    Fp a_bpc = fp_mul(a, b_plus_c);
    Fp ab2 = fp_mul(a, b);
    Fp ac = fp_mul(a, c);
    Fp ab_plus_ac = fp_add(ab2, ac);
    ASSERT_FP_EQ(a_bpc, ab_plus_ac, "a*(b+c) = a*b + a*c");
}

void test_fp_sqr() {
    Fp a = fp_from_u64(17);
    Fp expected = fp_from_u64(289); // 17^2

    Fp sq = fp_sqr(a);
    ASSERT_FP_EQ(sq, expected, "17^2 = 289");

    // sqr(a) = mul(a, a)
    Fp mul_aa = fp_mul(a, a);
    ASSERT_FP_EQ(sq, mul_aa, "sqr(a) = mul(a,a)");
}

void test_fp_inv() {
    // inv(1) = 1
    Fp inv_one = fp_inv(fp_mont_one());
    ASSERT_FP_EQ(inv_one, fp_mont_one(), "inv(1) = 1");

    // a * inv(a) = 1
    Fp a = fp_from_u64(42);
    Fp inv_a = fp_inv(a);
    Fp prod = fp_mul(a, inv_a);
    ASSERT_FP_EQ(prod, fp_mont_one(), "42 * inv(42) = 1");

    // Larger value
    Fp b = fp_from_u64(123456789);
    Fp inv_b = fp_inv(b);
    Fp prod2 = fp_mul(b, inv_b);
    ASSERT_FP_EQ(prod2, fp_mont_one(), "123456789 * inv(123456789) = 1");
}

void test_fp_from_raw_roundtrip() {
    // Test that from_raw -> to_raw is identity
    uint32_t raw_in[8] = {0xdeadbeef, 0x12345678, 0x9abcdef0, 0x11111111,
                           0x22222222, 0x33333333, 0x44444444, 0x10000000};
    Fp a = fp_from_raw(raw_in);
    uint32_t raw_out[8];
    fp_to_raw(a, raw_out);

    bool match = true;
    for (int i = 0; i < 8; ++i) {
        if (raw_in[i] != raw_out[i]) { match = false; break; }
    }
    ASSERT_TRUE(match, "from_raw -> to_raw roundtrip");

    // Small value roundtrip
    uint32_t small_in[8] = {7, 0, 0, 0, 0, 0, 0, 0};
    Fp s = fp_from_raw(small_in);
    uint32_t small_out[8];
    fp_to_raw(s, small_out);
    bool small_match = true;
    for (int i = 0; i < 8; ++i) {
        if (small_in[i] != small_out[i]) { small_match = false; break; }
    }
    ASSERT_TRUE(small_match, "small value roundtrip (7)");
}

void test_fp_modular_reduction() {
    // Test that (p-1) + 1 = 0 (mod p)
    uint32_t p_minus_1[8] = {
        0x00000000, 0xffffffff, 0xfffe5bfe, 0x53bda402,
        0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753
    };
    Fp pm1 = fp_from_raw(p_minus_1);
    Fp result = fp_add(pm1, fp_mont_one());
    ASSERT_FP_EQ(result, fp_mont_zero(), "(p-1) + 1 = 0 mod p");

    // p-1 should be -1, so (p-1)^2 = 1
    Fp neg_one_sq = fp_sqr(pm1);
    ASSERT_FP_EQ(neg_one_sq, fp_mont_one(), "(-1)^2 = 1");
}

void test_fr_basic() {
    // 0 + 0 = 0
    Fr z = fr_add(fr_zero(), fr_zero());
    ASSERT_TRUE(fr_eq(z, fr_zero()), "Fr: 0 + 0 = 0");

    // 1 * 1 = 1
    Fr one_sq = fr_mul(fr_one(), fr_one());
    ASSERT_TRUE(fr_eq(one_sq, fr_one()), "Fr: 1 * 1 = 1");

    // 1 - 1 = 0
    Fr one_sub = fr_sub(fr_one(), fr_one());
    ASSERT_TRUE(fr_eq(one_sub, fr_zero()), "Fr: 1 - 1 = 0");

    // Small mul: 6 * 7 = 42
    Fr six = fr_from_u64(6);
    Fr seven = fr_from_u64(7);
    Fr forty_two = fr_from_u64(42);
    Fr prod = fr_mul(six, seven);
    ASSERT_TRUE(fr_eq(prod, forty_two), "Fr: 6 * 7 = 42");

    // Inverse: a * inv(a) = 1
    Fr a = fr_from_u64(12345);
    Fr inv_a = fr_inv(a);
    Fr check = fr_mul(a, inv_a);
    ASSERT_TRUE(fr_eq(check, fr_one()), "Fr: a * inv(a) = 1");
}

void test_fr_to_raw_roundtrip() {
    uint32_t raw_in[8] = {0x11111111, 0x22222222, 0x00000000, 0x00000000,
                           0x00000000, 0x00000000, 0x00000000, 0x00000000};
    Fr a = fr_from_raw(raw_in);
    uint32_t raw_out[8];
    fr_to_raw(a, raw_out);

    bool match = true;
    for (int i = 0; i < 8; ++i) {
        if (raw_in[i] != raw_out[i]) { match = false; break; }
    }
    ASSERT_TRUE(match, "Fr: from_raw -> to_raw roundtrip");
}

void test_cuda_literal_constant_accessors() {
    // The arithmetic headers use literal accessors in __host__ __device__ code:
    // CUDA may not read unannotated namespace-scope arrays from device code.
    // Keep the device-safe literals locked to the canonical host test constants.
    bool fp_modulus_matches = true;
    bool fp_r2_matches = true;
    bool fr_modulus_matches = true;
    bool fr_r2_matches = true;
    for (int i = 0; i < 8; ++i) {
        fp_modulus_matches &= fp_modulus_limb(i) == FP_MODULUS[i];
        fp_r2_matches &= fp_r2_limb(i) == FP_R2[i];
        fr_modulus_matches &= fr_modulus_limb(i) == FR_MODULUS[i];
        fr_r2_matches &= fr_r2_limb(i) == FR_R2[i];
    }
    ASSERT_TRUE(fp_modulus_matches, "Fp device-safe modulus literals match canonical constants");
    ASSERT_TRUE(fp_r2_matches, "Fp device-safe R^2 literals match canonical constants");
    ASSERT_TRUE(fr_modulus_matches, "Fr device-safe modulus literals match canonical constants");
    ASSERT_TRUE(fr_r2_matches, "Fr device-safe R^2 literals match canonical constants");
}

Fp fp_from_vector_hex(const std::string& hex) {
    uint32_t limbs[8];
    cuda_verkle::test_util::load_hex_to_limbs(hex, limbs);
    return fp_from_raw(limbs);
}

void test_fp_rust_reference_vectors() {
    using namespace cuda_verkle::test_util;
    const JsonValue vectors = read_json(vector_path("field_test_vectors.json"));
    const JsonValue& cases = vectors.at("test_cases");
    ASSERT_TRUE(cases.type == JsonValue::Type::Array && !cases.array.empty(),
                "Rust field vector file contains test cases");

    for (size_t i = 0; i < cases.array.size(); ++i) {
        const JsonValue& test_case = cases.array[i];
        const std::string& op = test_case.at("op").as_string();
        const Fp a = fp_from_vector_hex(test_case.at("a").as_string());
        const Fp expected = fp_from_vector_hex(test_case.at("result").as_string());
        Fp actual = fp_mont_zero();
        if (op == "add") actual = fp_add(a, fp_from_vector_hex(test_case.at("b").as_string()));
        else if (op == "sub") actual = fp_sub(a, fp_from_vector_hex(test_case.at("b").as_string()));
        else if (op == "mul") actual = fp_mul(a, fp_from_vector_hex(test_case.at("b").as_string()));
        else if (op == "sqr") actual = fp_sqr(a);
        else if (op == "inv") actual = fp_inv(a);
        else throw std::runtime_error("Unknown Rust field vector operation: " + op);

        const std::string label = "Rust Fq vector " + std::to_string(i) + " (" + op + ")";
        ASSERT_FP_EQ(actual, expected, label.c_str());
    }
}

int main() {
    printf("Field Arithmetic Tests (Fp and Fr)\n");

    test_fp_zero_one();
    test_fp_add_sub();
    test_fp_mul();
    test_fp_sqr();
    test_fp_inv();
    test_fp_from_raw_roundtrip();
    test_fp_modular_reduction();

    test_fr_basic();
    test_fr_to_raw_roundtrip();
    test_cuda_literal_constant_accessors();
    test_fp_rust_reference_vectors();

    printf("\nResults: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
