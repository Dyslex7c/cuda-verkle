#include <cstdio>
#include <cstdint>
#include <cstring>
#include "../src/field/fp.cuh"
#include "../src/field/fr.cuh"
#include "../src/curve/bandersnatch.cuh"
#include "../src/curve/banderwagon.cuh"
#include "../src/constants/crs_points.cuh"
#include "../src/msm/msm_kernel.cuh"

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_TRUE(cond, msg) do { \
    if (cond) { tests_passed++; printf("  PASS: %s\n", msg); } \
    else { tests_failed++; printf("  FAIL: %s\n", msg); } \
} while(0)

void test_msm_all_zeros() {
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    Fr scalars[256];
    for (int i = 0; i < 256; ++i) scalars[i] = FR_ZERO;

    PointExtended result = msm_compute(scalars, crs_pts.x, crs_pts.y, 256);
    ASSERT_TRUE(point_is_identity(result), "MSM(all zeros) = identity");
}

void test_msm_single_one() {
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    Fr scalars[256];
    for (int i = 0; i < 256; ++i) scalars[i] = FR_ZERO;
    scalars[0] = fr_from_u64(1);

    PointExtended result = msm_compute(scalars, crs_pts.x, crs_pts.y, 256);
    PointAffine g0_aff = {crs_pts.x[0], crs_pts.y[0]};
    PointExtended G0 = point_from_affine(g0_aff);

    BanderwagonElement bw_result = {result};
    BanderwagonElement bw_g0 = {G0};
    ASSERT_TRUE(bw_eq(bw_result, bw_g0), "MSM(1 at idx 0) = G_0");
}

void test_msm_single_at_index() {
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    Fr scalars[256];
    for (int i = 0; i < 256; ++i) scalars[i] = FR_ZERO;
    scalars[5] = fr_from_u64(1);

    PointExtended result = msm_compute(scalars, crs_pts.x, crs_pts.y, 256);
    PointAffine g5_aff = {crs_pts.x[5], crs_pts.y[5]};
    PointExtended G5 = point_from_affine(g5_aff);

    BanderwagonElement bw_result = {result};
    BanderwagonElement bw_g5 = {G5};
    ASSERT_TRUE(bw_eq(bw_result, bw_g5), "MSM(1 at idx 5) = G_5");
}

void test_msm_scalar_two() {
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    Fr scalars[256];
    for (int i = 0; i < 256; ++i) scalars[i] = FR_ZERO;
    scalars[0] = fr_from_u64(2);

    PointExtended result = msm_compute(scalars, crs_pts.x, crs_pts.y, 256);

    PointAffine g0_aff = {crs_pts.x[0], crs_pts.y[0]};
    PointExtended G0 = point_from_affine(g0_aff);
    PointExtended expected = point_double(G0);

    BanderwagonElement bw_result = {result};
    BanderwagonElement bw_expected = {expected};
    ASSERT_TRUE(bw_eq(bw_result, bw_expected), "MSM(2 at idx 0) = 2*G_0");
}

void test_msm_pippenger_vs_naive() {
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    // Test with a few non-zero scalars
    Fr scalars[256];
    for (int i = 0; i < 256; ++i) scalars[i] = FR_ZERO;
    scalars[0] = fr_from_u64(17);
    scalars[1] = fr_from_u64(42);
    scalars[3] = fr_from_u64(100);
    scalars[10] = fr_from_u64(999);

    PointExtended pippenger_result = msm_compute(scalars, crs_pts.x, crs_pts.y, 256);
    PointExtended naive_result = msm_cpu_reference(scalars, crs_pts.x, crs_pts.y, 256);

    BanderwagonElement bw_pip = {pippenger_result};
    BanderwagonElement bw_naive = {naive_result};
    ASSERT_TRUE(bw_eq(bw_pip, bw_naive), "Pippenger matches naive (sparse inputs)");
}

void test_msm_all_ones() {
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    Fr scalars[256];
    for (int i = 0; i < 256; ++i) scalars[i] = fr_from_u64(1);

    PointExtended pippenger_result = msm_compute(scalars, crs_pts.x, crs_pts.y, 256);
    PointExtended naive_result = msm_cpu_reference(scalars, crs_pts.x, crs_pts.y, 256);

    BanderwagonElement bw_pip = {pippenger_result};
    BanderwagonElement bw_naive = {naive_result};
    ASSERT_TRUE(bw_eq(bw_pip, bw_naive), "Pippenger matches naive (all ones)");

    // Verify sum of all G_i is not identity
    ASSERT_TRUE(!point_is_identity(pippenger_result), "Sum of all G_i is not identity");
}

void test_msm_sequential() {
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    Fr scalars[256];
    for (int i = 0; i < 256; ++i) scalars[i] = fr_from_u64(i + 1);

    PointExtended pippenger_result = msm_compute(scalars, crs_pts.x, crs_pts.y, 256);
    PointExtended naive_result = msm_cpu_reference(scalars, crs_pts.x, crs_pts.y, 256);

    BanderwagonElement bw_pip = {pippenger_result};
    BanderwagonElement bw_naive = {naive_result};
    ASSERT_TRUE(bw_eq(bw_pip, bw_naive), "Pippenger matches naive (sequential 1..256)");
}

void test_commitment_serialization() {
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    Fr scalars[256];
    for (int i = 0; i < 256; ++i) scalars[i] = fr_from_u64(i + 1);

    PointExtended result = msm_compute(scalars, crs_pts.x, crs_pts.y, 256);
    BanderwagonElement bw = {result};

    uint8_t bytes[32];
    bw_to_bytes(bw, bytes);

    bool all_zero = true;
    for (int i = 0; i < 32; ++i) {
        if (bytes[i] != 0) { all_zero = false; break; }
    }
    ASSERT_TRUE(!all_zero, "Commitment serialization is non-zero");

    printf("  Commitment(1..256): 0x");
    for (int i = 0; i < 32; ++i) printf("%02x", bytes[i]);
    printf("\n");
}

int main() {
    printf("MSM and Commitment Tests (Phase 2)\n");

    test_msm_all_zeros();
    test_msm_single_one();
    test_msm_single_at_index();
    test_msm_scalar_two();
    test_msm_pippenger_vs_naive();
    test_msm_all_ones();
    test_msm_sequential();
    test_commitment_serialization();

    printf("\nResults: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
