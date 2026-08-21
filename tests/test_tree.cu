#include <cstdio>
#include <cstdint>
#include <cstring>
#include "../src/field/fp.cuh"
#include "../src/field/fr.cuh"
#include "../src/curve/bandersnatch.cuh"
#include "../src/curve/banderwagon.cuh"
#include "../src/constants/crs_points.cuh"
#include "../src/msm/msm_kernel.cuh"
#include "../src/msm/msm_kernel.cu"
#include "../src/tree/verkle_tree.cuh"

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_TRUE(cond, msg) do { \
    if (cond) { tests_passed++; printf("  PASS: %s\n", msg); } \
    else { tests_failed++; printf("  FAIL: %s\n", msg); } \
} while(0)

void test_empty_tree() {
    printf("\n--- test_empty_tree ---\n");

    VerkleTree tree;
    tree.init(1);
    tree.recompute_full();

    ASSERT_TRUE(point_is_identity(tree.root.point), "empty tree root = identity");

    uint8_t bytes[32];
    tree.get_root_bytes(bytes);
    bool all_zero = true;
    for (int i = 0; i < 32; ++i) {
        if (bytes[i] != 0) { all_zero = false; break; }
    }
    ASSERT_TRUE(all_zero, "empty tree root serializes to zero");

    tree.cleanup();
}

void test_tree_matches_msm() {
    printf("\n--- test_tree_matches_msm ---\n");

    VerkleTree tree;
    tree.init(1);

    Fr values[256];
    for (int i = 0; i < 256; ++i) values[i] = fr_from_u64(i + 1);
    tree.set_leaves(values, 256);

    // Compute the same MSM directly
    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);
    PointExtended expected = msm_compute(values, crs_pts.x, crs_pts.y, 256);

    BanderwagonElement bw_tree = {tree.root.point};
    BanderwagonElement bw_direct = {expected};
    ASSERT_TRUE(bw_eq(bw_tree, bw_direct), "tree root matches direct MSM");

    tree.cleanup();
}

void test_incremental_single_update() {
    printf("\n--- test_incremental_single_update ---\n");

    // Build tree with initial values
    VerkleTree tree_inc, tree_full;
    tree_inc.init(1);
    tree_full.init(1);

    Fr values[256];
    for (int i = 0; i < 256; ++i) values[i] = fr_from_u64(i + 1);
    tree_inc.set_leaves(values, 256);
    tree_full.set_leaves(values, 256);

    // Apply single update incrementally
    LeafUpdate update = {0, fr_from_u64(999)};
    tree_inc.apply_updates_incremental(&update, 1);

    // Apply same update via full recompute
    tree_full.apply_updates_full(&update, 1);

    BanderwagonElement bw_inc = {tree_inc.root.point};
    BanderwagonElement bw_full = {tree_full.root.point};
    ASSERT_TRUE(bw_eq(bw_inc, bw_full), "incremental single update matches full recompute");

    tree_inc.cleanup();
    tree_full.cleanup();
}

void test_incremental_multi_update() {
    printf("\n--- test_incremental_multi_update ---\n");

    VerkleTree tree_inc, tree_full;
    tree_inc.init(1);
    tree_full.init(1);

    Fr values[256];
    for (int i = 0; i < 256; ++i) values[i] = fr_from_u64(i * 7 + 3);
    tree_inc.set_leaves(values, 256);
    tree_full.set_leaves(values, 256);

    // Apply batch of updates
    LeafUpdate updates[5] = {
        {0,   fr_from_u64(1000)},
        {10,  fr_from_u64(2000)},
        {100, fr_from_u64(3000)},
        {200, fr_from_u64(4000)},
        {255, fr_from_u64(5000)},
    };
    tree_inc.apply_updates_incremental(updates, 5);
    tree_full.apply_updates_full(updates, 5);

    BanderwagonElement bw_inc = {tree_inc.root.point};
    BanderwagonElement bw_full = {tree_full.root.point};
    ASSERT_TRUE(bw_eq(bw_inc, bw_full), "incremental batch update matches full recompute");

    tree_inc.cleanup();
    tree_full.cleanup();
}

void test_incremental_all_leaves() {
    printf("\n--- test_incremental_all_leaves ---\n");

    VerkleTree tree_inc, tree_full;
    tree_inc.init(1);
    tree_full.init(1);

    Fr values[256];
    for (int i = 0; i < 256; ++i) values[i] = fr_from_u64(1);
    tree_inc.set_leaves(values, 256);
    tree_full.set_leaves(values, 256);

    // Update every leaf
    LeafUpdate updates[256];
    for (int i = 0; i < 256; ++i) {
        updates[i] = {i, fr_from_u64((i + 1) * 100)};
    }
    tree_inc.apply_updates_incremental(updates, 256);
    tree_full.apply_updates_full(updates, 256);

    BanderwagonElement bw_inc = {tree_inc.root.point};
    BanderwagonElement bw_full = {tree_full.root.point};
    ASSERT_TRUE(bw_eq(bw_inc, bw_full), "incremental all-leaf update matches full recompute");

    tree_inc.cleanup();
    tree_full.cleanup();
}

void test_depth_two_incremental_updates() {
    printf("\n--- test_depth_two_incremental_updates ---\n");

    VerkleTree tree_inc, tree_full;
    tree_inc.init(2);
    tree_full.init(2);

    // init() creates an all-zero tree.
    tree_inc.recompute_full();
    tree_full.recompute_full();

    LeafUpdate updates[4] = {
        {0,     fr_from_u64(11)},
        {255,   fr_from_u64(22)},
        {256,   fr_from_u64(33)},
        {65535, fr_from_u64(44)},
    };
    tree_inc.apply_updates_incremental(updates, 4);
    tree_full.apply_updates_full(updates, 4);

    BanderwagonElement bw_inc = {tree_inc.root.point};
    BanderwagonElement bw_full = {tree_full.root.point};
    ASSERT_TRUE(bw_eq(bw_inc, bw_full),
                "depth-2 incremental root update matches full recompute");
}

int main() {
    printf("========================================\n");
    printf(" Verkle Tree Tests (Phase 3)\n");
    printf("========================================\n");

    test_empty_tree();
    test_tree_matches_msm();
    test_incremental_single_update();
    test_incremental_multi_update();
    test_incremental_all_leaves();
    test_depth_two_incremental_updates();

    printf("\n========================================\n");
    printf(" Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
