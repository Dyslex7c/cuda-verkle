#include <cstdio>
#include <cstdint>
#include <cstring>
#include "../src/field/fp.cuh"
#include "../src/field/fr.cuh"
#include "../src/curve/bandersnatch.cuh"
#include "../src/curve/banderwagon.cuh"
#include "../src/constants/crs_points.cuh"

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_TRUE(cond, msg) do { \
    if (cond) { tests_passed++; printf("  PASS: %s\n", msg); } \
    else { tests_failed++; printf("  FAIL: %s\n", msg); } \
} while(0)

#define ASSERT_BW_EQ(a, b, msg) do { \
    if (bw_eq(a, b)) { tests_passed++; printf("  PASS: %s\n", msg); } \
    else { tests_failed++; printf("  FAIL: %s\n", msg); } \
} while(0)

static void print_point_affine(const char* label, const PointExtended& p) {
    PointAffine af = point_to_affine(p);
    uint32_t xr[8], yr[8];
    fp_to_raw(af.x, xr);
    fp_to_raw(af.y, yr);
    printf("  %s:\n    x = 0x", label);
    for (int i = 7; i >= 0; --i) printf("%08x", xr[i]);
    printf("\n    y = 0x");
    for (int i = 7; i >= 0; --i) printf("%08x", yr[i]);
    printf("\n");
}

static void limbs_to_big_endian(const uint32_t limbs[8], uint8_t out[32]) {
    for (int i = 0; i < 8; ++i) {
        const uint32_t limb = limbs[7 - i];
        out[i * 4] = static_cast<uint8_t>(limb >> 24);
        out[i * 4 + 1] = static_cast<uint8_t>(limb >> 16);
        out[i * 4 + 2] = static_cast<uint8_t>(limb >> 8);
        out[i * 4 + 3] = static_cast<uint8_t>(limb);
    }
}

void test_identity() {
    PointExtended id = point_identity();
    ASSERT_TRUE(point_is_identity(id), "identity is identity");

    // Adding identity to identity gives identity
    PointExtended id2 = point_add(id, id);
    ASSERT_TRUE(point_is_identity(id2), "id + id = id");
}

void test_point_add_properties() {
    // Load two CRS points as test points
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    PointAffine g0_aff = {crs_pts.x[0], crs_pts.y[0]};
    PointAffine g1_aff = {crs_pts.x[1], crs_pts.y[1]};
    PointExtended G0 = point_from_affine(g0_aff);
    PointExtended G1 = point_from_affine(g1_aff);

    // P + identity = P
    PointExtended id = point_identity();
    PointExtended g0_plus_id = point_add(G0, id);
    BanderwagonElement bw_g0 = {G0};
    BanderwagonElement bw_g0_plus_id = {g0_plus_id};
    ASSERT_BW_EQ(bw_g0, bw_g0_plus_id, "G0 + identity = G0");

    // Commutativity: G0 + G1 = G1 + G0
    PointExtended sum1 = point_add(G0, G1);
    PointExtended sum2 = point_add(G1, G0);
    BanderwagonElement bw_s1 = {sum1};
    BanderwagonElement bw_s2 = {sum2};
    ASSERT_BW_EQ(bw_s1, bw_s2, "G0 + G1 = G1 + G0 (commutativity)");

    // P + (-P) = identity
    PointExtended neg_g0 = point_neg(G0);
    PointExtended should_be_id = point_add(G0, neg_g0);
    ASSERT_TRUE(point_is_identity(should_be_id), "G0 + (-G0) = identity");
}

void test_double_consistency() {
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    PointAffine g0_aff = {crs_pts.x[0], crs_pts.y[0]};
    PointExtended G0 = point_from_affine(g0_aff);

    // double(G0) should equal add(G0, G0)
    PointExtended dbl = point_double(G0);
    PointExtended add = point_add(G0, G0);

    BanderwagonElement bw_dbl = {dbl};
    BanderwagonElement bw_add = {add};
    ASSERT_BW_EQ(bw_dbl, bw_add, "double(G0) = add(G0, G0)");
}

void test_scalar_mul() {
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    PointAffine g0_aff = {crs_pts.x[0], crs_pts.y[0]};
    PointExtended G0 = point_from_affine(g0_aff);

    // scalar_mul(G0, 1) = G0
    Fr one = fr_from_u64(1);
    PointExtended g0_times_1 = scalar_mul(G0, one);
    BanderwagonElement bw_g0 = {G0};
    BanderwagonElement bw_g0x1 = {g0_times_1};
    ASSERT_BW_EQ(bw_g0, bw_g0x1, "G0 * 1 = G0");

    // scalar_mul(G0, 2) = double(G0)
    Fr two = fr_from_u64(2);
    PointExtended g0_times_2 = scalar_mul(G0, two);
    PointExtended g0_dbl = point_double(G0);
    BanderwagonElement bw_g0x2 = {g0_times_2};
    BanderwagonElement bw_g0d = {g0_dbl};
    ASSERT_BW_EQ(bw_g0x2, bw_g0d, "G0 * 2 = double(G0)");

    // scalar_mul(G0, 3) = G0 + G0 + G0
    Fr three = fr_from_u64(3);
    PointExtended g0_times_3 = scalar_mul(G0, three);
    PointExtended g0_triple = point_add(g0_dbl, G0);
    BanderwagonElement bw_g0x3 = {g0_times_3};
    BanderwagonElement bw_g0t = {g0_triple};
    ASSERT_BW_EQ(bw_g0x3, bw_g0t, "G0 * 3 = G0 + G0 + G0");

    // scalar_mul(G0, 0) = identity
    Fr zero = FR_ZERO;
    PointExtended g0_times_0 = scalar_mul(G0, zero);
    ASSERT_TRUE(point_is_identity(g0_times_0), "G0 * 0 = identity");
}

void test_associativity() {
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    PointAffine g0_aff = {crs_pts.x[0], crs_pts.y[0]};
    PointAffine g1_aff = {crs_pts.x[1], crs_pts.y[1]};
    PointAffine g2_aff = {crs_pts.x[2], crs_pts.y[2]};
    PointExtended G0 = point_from_affine(g0_aff);
    PointExtended G1 = point_from_affine(g1_aff);
    PointExtended G2 = point_from_affine(g2_aff);

    // (G0 + G1) + G2 = G0 + (G1 + G2)
    PointExtended lhs = point_add(point_add(G0, G1), G2);
    PointExtended rhs = point_add(G0, point_add(G1, G2));
    BanderwagonElement bw_lhs = {lhs};
    BanderwagonElement bw_rhs = {rhs};
    ASSERT_BW_EQ(bw_lhs, bw_rhs, "(G0+G1)+G2 = G0+(G1+G2) (associativity)");
}

void test_banderwagon_equality() {
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    PointAffine g0_aff = {crs_pts.x[0], crs_pts.y[0]};
    PointExtended G0 = point_from_affine(g0_aff);

    // A point equals itself
    BanderwagonElement bw1 = {G0};
    BanderwagonElement bw2 = {G0};
    ASSERT_TRUE(bw_eq(bw1, bw2), "point == itself");

    // A point and its double are NOT equal
    PointExtended G0_dbl = point_double(G0);
    BanderwagonElement bw3 = {G0_dbl};
    ASSERT_TRUE(!bw_eq(bw1, bw3), "G0 != 2*G0");

    // In Banderwagon: P == -(-P)
    PointExtended neg_g0 = point_neg(G0);
    PointExtended neg_neg_g0 = point_neg(neg_g0);
    BanderwagonElement bw4 = {neg_neg_g0};
    ASSERT_TRUE(bw_eq(bw1, bw4), "P == neg(neg(P))");
}

void test_banderwagon_serialization() {
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    PointAffine g0_aff = {crs_pts.x[0], crs_pts.y[0]};
    PointExtended G0 = point_from_affine(g0_aff);
    BanderwagonElement bw_g0 = {G0};

    // Serialize and check it produces 32 non-zero bytes
    uint8_t bytes[32];
    bw_to_bytes(bw_g0, bytes);

    bool all_zero = true;
    for (int i = 0; i < 32; ++i) {
        if (bytes[i] != 0) { all_zero = false; break; }
    }
    ASSERT_TRUE(!all_zero, "serialization produces non-zero bytes");

    // Print the serialized form for manual verification
    printf("  G0 serialized: 0x");
    for (int i = 0; i < 32; ++i) printf("%02x", bytes[i]);
    printf("\n");

    // Serialize identity
    BanderwagonElement bw_id = bw_identity();
    uint8_t id_bytes[32];
    bw_to_bytes(bw_id, id_bytes);
    printf("  Identity serialized: 0x");
    for (int i = 0; i < 32; ++i) printf("%02x", id_bytes[i]);
    printf("\n");
}

void test_strict_scalar_decoding() {
    const Fr inputs[] = {
        FR_ZERO,
        FR_ONE,
        fr_from_u64(42),
        fr_from_u64(0xffffffffULL),
    };
    bool all_roundtrip = true;
    for (const Fr& input : inputs) {
        uint8_t bytes[32];
        Fr decoded;
        fr_to_bytes(input, bytes);
        all_roundtrip = all_roundtrip && fr_from_bytes_strict(bytes, decoded) && fr_eq(input, decoded);
    }
    ASSERT_TRUE(all_roundtrip, "strict scalar decoding round-trips canonical inputs");

    uint8_t modulus[32];
    limbs_to_big_endian(FR_MODULUS, modulus);
    Fr ignored;
    ASSERT_TRUE(!fr_from_bytes_strict(modulus, ignored), "strict scalar decoder rejects subgroup order");

    uint8_t all_ones[32];
    std::memset(all_ones, 0xff, sizeof(all_ones));
    ASSERT_TRUE(!fr_from_bytes_strict(all_ones, ignored), "strict scalar decoder rejects non-canonical all-ones input");
}

void test_strict_banderwagon_decoding() {
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);
    const BanderwagonElement generator = {point_from_affine({crs_pts.x[0], crs_pts.y[0]})};
    uint8_t encoded[32];
    bw_to_bytes(generator, encoded);

    BanderwagonElement decoded;
    uint8_t reencoded[32];
    const bool decoded_generator = bw_from_bytes_strict(encoded, decoded);
    if (decoded_generator) bw_to_bytes(decoded, reencoded);
    ASSERT_TRUE(decoded_generator && bw_eq(generator, decoded) && std::memcmp(encoded, reencoded, 32) == 0,
                "strict Banderwagon decoder round-trips Rust-compatible generator bytes");

    bool sampled_crs_roundtrip = true;
    for (int i = 0; i < 256; i += 17) {
        const BanderwagonElement source = {point_from_affine({crs_pts.x[i], crs_pts.y[i]})};
        uint8_t source_bytes[32];
        uint8_t checked_bytes[32];
        bw_to_bytes(source, source_bytes);
        if (!bw_from_bytes_strict(source_bytes, decoded)) {
            sampled_crs_roundtrip = false;
            break;
        }
        bw_to_bytes(decoded, checked_bytes);
        sampled_crs_roundtrip = sampled_crs_roundtrip && bw_eq(source, decoded) &&
                                std::memcmp(source_bytes, checked_bytes, 32) == 0;
    }
    ASSERT_TRUE(sampled_crs_roundtrip, "strict Banderwagon decoder round-trips sampled CRS encodings");

    uint8_t identity[32] = {0};
    BanderwagonElement decoded_identity;
    ASSERT_TRUE(bw_from_bytes_strict(identity, decoded_identity) && bw_eq(decoded_identity, bw_identity()),
                "strict Banderwagon decoder accepts the canonical identity encoding");

    uint8_t field_modulus[32];
    limbs_to_big_endian(FP_MODULUS, field_modulus);
    ASSERT_TRUE(!bw_from_bytes_strict(field_modulus, decoded),
                "strict Banderwagon decoder rejects non-canonical x encoding");

    bool off_curve_rejected = false;
    bool subgroup_rejected = false;
    for (uint64_t candidate = 1; candidate < 4096 && (!off_curve_rejected || !subgroup_rejected); ++candidate) {
        const Fp x = fp_from_u64(candidate);
        Fp y;
        uint8_t bytes[32];
        fp_to_bytes(x, bytes);
        const bool curve_point = bw_recover_y_from_x(x, y);
        if (!curve_point) {
            off_curve_rejected = !bw_from_bytes_strict(bytes, decoded);
            continue;
        }
        const BanderwagonElement point = {{x, y, fp_mul(x, y), FP_MONT_ONE}};
        if (!bw_subgroup_check(point)) subgroup_rejected = !bw_from_bytes_strict(bytes, decoded);
    }
    ASSERT_TRUE(off_curve_rejected, "strict Banderwagon decoder rejects x values with no curve point");
    ASSERT_TRUE(subgroup_rejected, "strict Banderwagon decoder rejects valid-curve non-subgroup points");
}

void test_crs_loading() {
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    ASSERT_TRUE(crs_pts.loaded, "CRS loaded successfully");

    // Check that no CRS point is the identity
    for (int i = 0; i < 10; ++i) {
        PointAffine aff = {crs_pts.x[i], crs_pts.y[i]};
        PointExtended pt = point_from_affine(aff);
        ASSERT_TRUE(!point_is_identity(pt), "CRS point is not identity");
    }

    // Check that CRS points satisfy the curve equation: ax^2 + y^2 = 1 + dx^2y^2
    for (int i = 0; i < 10; ++i) {
        Fp x = crs_pts.x[i];
        Fp y = crs_pts.y[i];
        Fp x2 = fp_sqr(x);
        Fp y2 = fp_sqr(y);
        Fp lhs = fp_add(fp_mul(COEFF_A, x2), y2);                 // ax^2 + y^2
        Fp rhs = fp_add(FP_MONT_ONE, fp_mul(COEFF_D, fp_mul(x2, y2))); // 1 + dx^2y^2
        ASSERT_TRUE(fp_eq(lhs, rhs), "CRS point on curve");
    }
}

void test_scalar_distributivity() {
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    PointAffine g0_aff = {crs_pts.x[0], crs_pts.y[0]};
    PointExtended G0 = point_from_affine(g0_aff);

    // (a + b) * G = a*G + b*G
    Fr a = fr_from_u64(17);
    Fr b = fr_from_u64(29);
    Fr a_plus_b = fr_add(a, b);

    PointExtended lhs = scalar_mul(G0, a_plus_b);
    PointExtended aG = scalar_mul(G0, a);
    PointExtended bG = scalar_mul(G0, b);
    PointExtended rhs = point_add(aG, bG);

    BanderwagonElement bw_lhs = {lhs};
    BanderwagonElement bw_rhs = {rhs};
    ASSERT_BW_EQ(bw_lhs, bw_rhs, "(17+29)*G = 17*G + 29*G");
}

int main() {
    printf("Curve Arithmetic Tests (Bandersnatch / Banderwagon)\n");

    test_identity();
    test_point_add_properties();
    test_double_consistency();
    test_scalar_mul();
    test_associativity();
    test_banderwagon_equality();
    test_banderwagon_serialization();
    test_strict_scalar_decoding();
    test_strict_banderwagon_decoding();
    test_crs_loading();
    test_scalar_distributivity();

    printf("\nResults: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
