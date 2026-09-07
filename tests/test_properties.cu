#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

// Include the test-only parser before field headers: <cstdlib>, which the
// parser needs, may expose the C FP_ZERO macro on some host toolchains.
#include "../src/util/test_vectors.cuh"
#include "../src/field/fp.cuh"
#include "../src/field/fr.cuh"
#include "../src/curve/bandersnatch.cuh"
#include "../src/curve/banderwagon.cuh"
#include "../src/constants/crs_points.cuh"
#include "../src/msm/msm_kernel.cuh"
#include "../src/tree/verkle_tree.cuh"

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_TRUE(cond, msg) do { \
    if (cond) { ++tests_passed; std::printf("  PASS: %s\n", msg); } \
    else { ++tests_failed; std::printf("  FAIL: %s\n", msg); } \
} while (0)

// Fixed seed keeps failing inputs reproducible while still exercising all 256
// scalar bits. This is deliberately not std::rand(), whose sequence differs
// between platforms.
struct SplitMix64 {
    uint64_t state;

    explicit SplitMix64(uint64_t seed) : state(seed) {}

    uint64_t next() {
        uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }
};

static Fr random_full_width_scalar(SplitMix64& rng) {
    uint32_t limbs[8];
    for (int i = 0; i < 8; i += 2) {
        const uint64_t word = rng.next();
        limbs[i] = static_cast<uint32_t>(word);
        limbs[i + 1] = static_cast<uint32_t>(word >> 32);
    }
    return fr_from_raw(limbs);
}

static bool raw_less_than_fr_modulus(const uint32_t raw[8]) {
    for (int i = 7; i >= 0; --i) {
        if (raw[i] < FR_MODULUS[i]) return true;
        if (raw[i] > FR_MODULUS[i]) return false;
    }
    return false;
}

static bool throws_invalid_argument(const std::string& input) {
    try {
        cuda_verkle::test_util::JsonParser parser(input);
        (void)parser.parse();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

void test_full_width_scalar_properties() {
    SplitMix64 rng(0x7b1d5eedc0ffee42ULL);
    bool canonical = true;
    bool roundtrip = true;
    bool strict_decode_roundtrip = true;
    bool additive_inverse = true;
    bool multiplicative_inverse = true;

    for (int iteration = 0; iteration < 128; ++iteration) {
        const Fr a = random_full_width_scalar(rng);
        const Fr b = random_full_width_scalar(rng);
        uint32_t raw[8];
        fr_to_raw(a, raw);
        canonical = canonical && raw_less_than_fr_modulus(raw);
        const Fr restored = fr_from_raw(raw);
        roundtrip = roundtrip && fr_eq(a, restored);
        uint8_t encoded[32];
        Fr decoded;
        fr_to_bytes(a, encoded);
        strict_decode_roundtrip = strict_decode_roundtrip && fr_from_bytes_strict(encoded, decoded) && fr_eq(a, decoded);
        additive_inverse = additive_inverse && fr_eq(fr_sub(fr_add(a, b), b), a);
        if (!fr_is_zero(a)) {
            multiplicative_inverse = multiplicative_inverse && fr_eq(fr_mul(a, fr_inv(a)), FR_ONE);
        }
    }

    // Exact modulus and all-one inputs are important reduction boundaries
    // which ordinary small-value tests never reach.
    uint32_t modulus[8];
    uint32_t all_ones[8];
    for (int i = 0; i < 8; ++i) {
        modulus[i] = FR_MODULUS[i];
        all_ones[i] = 0xffffffffU;
    }
    const Fr reduced_modulus = fr_from_raw(modulus);
    const Fr reduced_all_ones = fr_from_raw(all_ones);
    uint32_t all_ones_raw[8];
    fr_to_raw(reduced_all_ones, all_ones_raw);

    ASSERT_TRUE(canonical, "128 full-width scalar inputs reduce to canonical Fr values");
    ASSERT_TRUE(roundtrip, "full-width scalar serialization round-trips");
    ASSERT_TRUE(strict_decode_roundtrip, "full-width scalar strict decoding round-trips");
    ASSERT_TRUE(additive_inverse, "random Fr addition/subtraction property holds");
    ASSERT_TRUE(multiplicative_inverse, "random non-zero Fr inverse property holds");
    ASSERT_TRUE(fr_eq(reduced_modulus, FR_ZERO), "Fr modulus reduces to zero");
    ASSERT_TRUE(raw_less_than_fr_modulus(all_ones_raw), "all-ones scalar reduces to canonical Fr");
}

void test_curve_and_msm_differential_properties() {
    crs::CRSPoints crs_points;
    crs::load_crs(crs_points);
    SplitMix64 rng(0x5ca1ab1e9a11d00dULL);
    bool scalar_distributive = true;
    bool scalar_doubling = true;
    bool msm_matches_naive = true;

    for (int iteration = 0; iteration < 6; ++iteration) {
        const int point_index = static_cast<int>(rng.next() % MSM_SIZE);
        const PointExtended point = point_from_affine({crs_points.x[point_index], crs_points.y[point_index]});
        const Fr a = random_full_width_scalar(rng);
        const Fr b = random_full_width_scalar(rng);
        const PointExtended lhs = scalar_mul(point, fr_add(a, b));
        const PointExtended rhs = point_add(scalar_mul(point, a), scalar_mul(point, b));
        scalar_distributive = scalar_distributive && bw_eq({lhs}, {rhs});

        const PointExtended doubled_scalar = scalar_mul(point, fr_add(a, a));
        const PointExtended doubled_point = point_double(scalar_mul(point, a));
        scalar_doubling = scalar_doubling && bw_eq({doubled_scalar}, {doubled_point});

        Fr scalars[MSM_SIZE];
        for (int i = 0; i < MSM_SIZE; ++i) scalars[i] = random_full_width_scalar(rng);
        const PointExtended pippenger = msm_compute(scalars, crs_points.x, crs_points.y, MSM_SIZE);
        const PointExtended naive = msm_cpu_reference(scalars, crs_points.x, crs_points.y, MSM_SIZE);
        msm_matches_naive = msm_matches_naive && bw_eq({pippenger}, {naive});
    }

    ASSERT_TRUE(scalar_distributive, "full-width scalar distributivity holds across CRS points");
    ASSERT_TRUE(scalar_doubling, "full-width scalar doubling holds across CRS points");
    ASSERT_TRUE(msm_matches_naive, "random full-width Pippenger MSMs match independent naive MSM");
}

void test_malformed_input_handling() {
    crs::CRSPoints crs_points;
    crs::load_crs(crs_points);
    Fr values[MSM_SIZE];
    for (int i = 0; i < MSM_SIZE; ++i) values[i] = fr_from_u64(static_cast<uint64_t>(i + 1));

    const PointExtended empty = msm_compute(nullptr, crs_points.x, crs_points.y, 0);
    const PointExtended negative = msm_compute(values, crs_points.x, crs_points.y, -4);
    const PointExtended width = msm_compute(values, crs_points.x, crs_points.y, MSM_SIZE);
    const PointExtended over_width = msm_compute(values, crs_points.x, crs_points.y, MSM_SIZE + 1);
    ASSERT_TRUE(point_is_identity(empty) && point_is_identity(negative),
                "MSM rejects zero and negative lengths without dereferencing inputs");
    ASSERT_TRUE(bw_eq({width}, {over_width}), "MSM clamps oversized public lengths to CRS width");

    Eip6800StateTree tree;
    VerkleKey tree_key{};
    VerkleValue tree_value{};
    tree_value[0] = 1;
    tree.set(tree_key, tree_value);
    const PointExtended original_root = tree.root();
    tree.erase(VerkleKey{});
    ASSERT_TRUE(!point_is_identity(original_root) && point_is_identity(tree.root()),
                "state tree erase restores an empty root without retaining stale state");

    uint32_t limbs[8];
    bool bad_hex_rejected = false;
    try {
        cuda_verkle::test_util::load_hex_to_limbs("0xzz", limbs);
    } catch (const std::invalid_argument&) {
        bad_hex_rejected = true;
    }
    ASSERT_TRUE(bad_hex_rejected, "vector hex loader rejects malformed hexadecimal input");
    ASSERT_TRUE(throws_invalid_argument("{\"test\":]"), "JSON parser rejects malformed object input");
    ASSERT_TRUE(throws_invalid_argument("[1, 2"), "JSON parser rejects truncated array input");

    // A compact deterministic malformed-input fuzz corpus guards parser error
    // paths without relying on a platform-specific fuzzing runtime.
    SplitMix64 rng(0x0badf00d12345678ULL);
    bool completed = true;
    for (int case_index = 0; case_index < 256; ++case_index) {
        std::string input;
        const int length = static_cast<int>(rng.next() % 48);
        for (int i = 0; i < length; ++i) input += static_cast<char>(rng.next() & 0x7fU);
        try {
            cuda_verkle::test_util::JsonParser parser(input);
            (void)parser.parse();
        } catch (const std::runtime_error&) {
            // Expected for malformed fuzz inputs.
        } catch (...) {
            completed = false;
        }
    }
    ASSERT_TRUE(completed, "256 deterministic malformed JSON fuzz inputs fail safely");
}

int main() {
    std::printf("Randomized, Property, Fuzz, and Differential Tests\n");
    test_full_width_scalar_properties();
    test_curve_and_msm_differential_properties();
    test_malformed_input_handling();
    std::printf("\nResults: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed == 0 ? 0 : 1;
}
