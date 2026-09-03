#include <cstdio>
#include <cstdint>
#include <cstring>
#include "../src/field/fp.cuh"
#include "../src/field/fr.cuh"
#include "../src/curve/bandersnatch.cuh"
#include "../src/curve/banderwagon.cuh"
#include "../src/constants/crs_points.cuh"
#include "../src/msm/msm_kernel.cuh"
#include "../src/tree/verkle_tree.cuh"
#include "../src/util/test_vectors.cuh"

#include <string>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_TRUE(cond, msg) do { \
    if (cond) { tests_passed++; printf("  PASS: %s\n", msg); } \
    else { tests_failed++; printf("  FAIL: %s\n", msg); } \
} while(0)

void test_empty_tree() {
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

Fr fr_from_tree_vector_hex(const std::string& hex) {
    uint32_t limbs[8];
    cuda_verkle::test_util::load_hex_to_limbs(hex, limbs);
    return fr_from_raw(limbs);
}

std::string tree_bytes_to_hex(const uint8_t bytes[32]) {
    static const char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (int i = 0; i < 32; ++i) {
        result += hex[bytes[i] >> 4];
        result += hex[bytes[i] & 0x0f];
    }
    return result;
}

void test_rust_tree_vectors() {
    using namespace cuda_verkle::test_util;
    const JsonValue vectors = read_json(vector_path("tree_test_vectors.json"));
    ASSERT_TRUE(vectors.at("tree_depth").as_size() == 1 && vectors.at("width").as_size() == 256,
                "Rust tree vector shape is depth 1 and width 256");
    const JsonValue& cases = vectors.at("test_cases");
    ASSERT_TRUE(cases.type == JsonValue::Type::Array && !cases.array.empty(),
                "Rust tree vector file contains test cases");

    for (size_t case_index = 0; case_index < cases.array.size(); ++case_index) {
        const JsonValue& test_case = cases.array[case_index];
        const JsonValue& initial_leaves = test_case.at("initial_leaves");
        const JsonValue& updates_json = test_case.at("updates");
        if (initial_leaves.type != JsonValue::Type::Array || initial_leaves.array.size() != 256 ||
            updates_json.type != JsonValue::Type::Array) {
            throw std::runtime_error("Invalid Rust tree vector shape");
        }

        Fr leaves[256];
        for (size_t i = 0; i < 256; ++i) leaves[i] = fr_from_tree_vector_hex(initial_leaves.array[i].as_string());
        std::vector<LeafUpdate> updates;
        updates.reserve(updates_json.array.size());
        for (const JsonValue& update : updates_json.array) {
            const size_t index = update.at("index").as_size();
            if (index >= 256) throw std::runtime_error("Rust tree update index out of range");
            updates.push_back({static_cast<int>(index), fr_from_tree_vector_hex(update.at("new_value").as_string())});
        }

        VerkleTree tree;
        tree.init(1);
        tree.set_leaves(leaves, 256);
        tree.apply_updates_incremental(updates.data(), static_cast<int>(updates.size()));
        uint8_t actual_root[32];
        tree.get_root_bytes(actual_root);
        const std::string label = "Rust tree vector " + test_case.at("name").as_string();
        ASSERT_TRUE(tree_bytes_to_hex(actual_root) == test_case.at("expected_root").as_string(), label.c_str());
        tree.cleanup();
    }
}

int main() {
    printf("Verkle Tree Tests (Phase 3)\n");

    test_empty_tree();
    test_tree_matches_msm();
    test_incremental_single_update();
    test_incremental_multi_update();
    test_incremental_all_leaves();
    test_depth_two_incremental_updates();
    test_rust_tree_vectors();

    printf("\nResults: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
