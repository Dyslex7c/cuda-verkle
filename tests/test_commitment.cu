#include <cstdio>
#include <cstdint>
#include <cstring>
#include "../src/field/fp.cuh"
#include "../src/field/fr.cuh"
#include "../src/curve/bandersnatch.cuh"
#include "../src/curve/banderwagon.cuh"
#include "../src/constants/crs_points.cuh"
#include "../src/msm/msm_kernel.cuh"
#include "../src/commitment/pedersen.cuh"

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_TRUE(cond, msg) do { \
    if (cond) { tests_passed++; printf("  PASS: %s\n", msg); } \
    else { tests_failed++; printf("  FAIL: %s\n", msg); } \
} while(0)

void test_commitment_identity() {
    PedersenCommitment pc;
    pc.init();

    // All-zero values should give identity commitment
    Fr values[256];
    for (int i = 0; i < 256; ++i) values[i] = FR_ZERO;

    BanderwagonElement c = pc.commit(values, 256);
    ASSERT_TRUE(point_is_identity(c.point), "commit(all zeros) = identity");
}

void test_commitment_single_basis() {
    PedersenCommitment pc;
    pc.init();

    // value=1 at index 0, rest zero -> commitment = G_0
    Fr values[256];
    for (int i = 0; i < 256; ++i) values[i] = FR_ZERO;
    values[0] = fr_from_u64(1);

    BanderwagonElement c = pc.commit(values, 256);
    PointAffine g0 = {pc.crs.x[0], pc.crs.y[0]};
    BanderwagonElement expected = {point_from_affine(g0)};
    ASSERT_TRUE(bw_eq(c, expected), "commit([1,0,...,0]) = G_0");
}

void test_commitment_to_bytes() {
    PedersenCommitment pc;
    pc.init();

    Fr values[256];
    for (int i = 0; i < 256; ++i) values[i] = fr_from_u64(i + 1);

    uint8_t bytes[32];
    pc.commit_to_bytes(values, 256, bytes);

    bool all_zero = true;
    for (int i = 0; i < 32; ++i) {
        if (bytes[i] != 0) { all_zero = false; break; }
    }
    ASSERT_TRUE(!all_zero, "commit_to_bytes produces non-zero output");

    printf("  Commitment bytes: 0x");
    for (int i = 0; i < 32; ++i) printf("%02x", bytes[i]);
    printf("\n");
}

void test_commitment_linearity() {
    PedersenCommitment pc;
    pc.init();

    // commit(a + b) == commit(a) + commit(b) for disjoint supports
    Fr a_values[256], b_values[256], ab_values[256];
    for (int i = 0; i < 256; ++i) {
        a_values[i] = FR_ZERO;
        b_values[i] = FR_ZERO;
        ab_values[i] = FR_ZERO;
    }
    a_values[0] = fr_from_u64(5);
    a_values[1] = fr_from_u64(10);
    b_values[2] = fr_from_u64(15);
    b_values[3] = fr_from_u64(20);
    ab_values[0] = fr_from_u64(5);
    ab_values[1] = fr_from_u64(10);
    ab_values[2] = fr_from_u64(15);
    ab_values[3] = fr_from_u64(20);

    BanderwagonElement ca = pc.commit(a_values, 256);
    BanderwagonElement cb = pc.commit(b_values, 256);
    BanderwagonElement cab = pc.commit(ab_values, 256);
    BanderwagonElement ca_plus_cb = bw_add(ca, cb);

    ASSERT_TRUE(bw_eq(cab, ca_plus_cb), "commit(a+b) = commit(a) + commit(b)");
}

int main() {
    printf("Pedersen Commitment Tests\n");

    test_commitment_identity();
    test_commitment_single_basis();
    test_commitment_to_bytes();
    test_commitment_linearity();

    printf("\nResults: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
