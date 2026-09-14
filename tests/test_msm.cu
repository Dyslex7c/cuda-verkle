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
    for (int i = 0; i < 256; ++i) scalars[i] = fr_zero();

    PointExtended result = msm_compute(scalars, crs_pts.x, crs_pts.y, 256);
    ASSERT_TRUE(point_is_identity(result), "MSM(all zeros) = identity");
}

void test_msm_single_one() {
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    Fr scalars[256];
    for (int i = 0; i < 256; ++i) scalars[i] = fr_zero();
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
    for (int i = 0; i < 256; ++i) scalars[i] = fr_zero();
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
    for (int i = 0; i < 256; ++i) scalars[i] = fr_zero();
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
    for (int i = 0; i < 256; ++i) scalars[i] = fr_zero();
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

// verifies the CUDA decomposition even on hosts without an NVIDIA GPU.
PointExtended msm_gpu_pippenger_model(
    const Fr scalars[], const Fp point_x[], const Fp point_y[], int n) {
    uint32_t raw_scalars[MSM_SIZE][8];
    for (int i = 0; i < n; ++i) fr_to_raw(scalars[i], raw_scalars[i]);

    PointExtended window_sums[MSM_NUM_WINDOWS];
    for (int window = 0; window < MSM_NUM_WINDOWS; ++window) {
        PointExtended buckets[MSM_BUCKET_COUNT];
        for (int bucket = 0; bucket < MSM_BUCKET_COUNT; ++bucket) {
            buckets[bucket] = point_identity();
        }

        const int bit_start = window * MSM_WINDOW_BITS;
        const int limb_index = bit_start / 32;
        const int bit_offset = bit_start % 32;
        for (int i = 0; i < n; ++i) {
            uint32_t digit = (raw_scalars[i][limb_index] >> bit_offset) &
                             (MSM_BUCKET_COUNT - 1);
            if (digit != 0) {
                PointAffine base = {point_x[i], point_y[i]};
                buckets[digit] = point_add(buckets[digit], point_from_affine(base));
            }
        }

        PointExtended running_sum = point_identity();
        window_sums[window] = point_identity();
        for (int bucket = MSM_BUCKET_COUNT - 1; bucket >= 1; --bucket) {
            running_sum = point_add(running_sum, buckets[bucket]);
            window_sums[window] = point_add(window_sums[window], running_sum);
        }
    }

    PointExtended total = window_sums[MSM_NUM_WINDOWS - 1];
    for (int window = MSM_NUM_WINDOWS - 2; window >= 0; --window) {
        for (int bit = 0; bit < MSM_WINDOW_BITS; ++bit) total = point_double(total);
        total = point_add(total, window_sums[window]);
    }
    return total;
}

void test_gpu_pippenger_window_model() {
    printf("\n--- test_gpu_pippenger_window_model ---\n");

    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    Fr scalars[MSM_SIZE];
    for (int i = 0; i < MSM_SIZE; ++i) {
        // Values near Fr's modulus exercise the highest Pippenger windows.
        scalars[i] = (i & 1) ? fr_sub(fr_zero(), fr_from_u64(i + 1))
                             : fr_from_u64(static_cast<uint64_t>(i + 1) * 1234567ULL);
    }

    PointExtended model = msm_gpu_pippenger_model(scalars, crs_pts.x, crs_pts.y, MSM_SIZE);
    PointExtended cpu = msm_compute(scalars, crs_pts.x, crs_pts.y, MSM_SIZE);
    ASSERT_TRUE(bw_eq({model}, {cpu}),
                "GPU Pippenger window model matches CPU Pippenger");
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
    test_gpu_pippenger_window_model();
    test_commitment_serialization();

    printf("\nResults: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
