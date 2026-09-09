#include <cstdio>
#include <cstring>
#include <string>
#include "../src/util/test_vectors.cuh"
#include "../src/tree/verkle_tree.cuh"

static int passed = 0, failed = 0;
#define CHECK(x, m) do { if (x) { ++passed; std::printf("  PASS: %s\n", m); } else { ++failed; std::printf("  FAIL: %s\n", m); } } while (0)

static VerkleKey key(uint8_t first, uint8_t second, uint8_t suffix) {
    VerkleKey result{}; result[0] = first; result[1] = second; result[31] = suffix; return result;
}
static VerkleValue value(uint8_t first) { VerkleValue result{}; result[0] = first; return result; }
static bool same(const PointExtended& a, const PointExtended& b) { return bw_eq({a}, {b}); }

static void hex32(const std::string& text, uint8_t out[32]) {
    for (int i = 0; i < 32; ++i) out[i] = static_cast<uint8_t>((cuda_verkle::test_util::hex_char_to_int(text[2 * i]) << 4) | cuda_verkle::test_util::hex_char_to_int(text[2 * i + 1]));
}

void test_empty_and_zero_value() {
    Eip6800StateTree tree;
    CHECK(point_is_identity(tree.root()), "empty EIP-6800 state tree has identity root");
    const VerkleKey k = key(1, 2, 3);
    tree.set(k, VerkleValue{});
    CHECK(!point_is_identity(tree.root()), "present all-zero value differs from absent key");
    tree.erase(k);
    CHECK(point_is_identity(tree.root()), "erasing the only key restores the empty root");
}

void test_determinism_and_branching() {
    const VerkleKey a = key(0x11, 0x22, 0x01);
    const VerkleKey b = key(0x11, 0x22, 0xfe);
    const VerkleKey c = key(0x99, 0x01, 0x07);
    const VerkleKey d = key(0x11, 0x23, 0x07);
    Eip6800StateTree left, right;
    left.set(a, value(7)); left.set(b, value(8)); left.set(c, value(9)); left.set(d, value(10));
    right.set(d, value(10)); right.set(c, value(9)); right.set(b, value(8)); right.set(a, value(7));
    CHECK(same(left.root(), right.root()), "root is independent of insertion order across stems and suffixes");
    const PointExtended before = left.root();
    left.set(b, value(42));
    CHECK(!same(before, left.root()), "updating a present value changes the root");

    const VerkleNode& root = left.root_node();
    CHECK(root.kind == VerkleNode::Kind::branch && root.children[0x11] && root.children[0x99],
          "root stores explicit branch children for divergent first stem bytes");
    const VerkleNode* branch = root.children[0x11].get();
    CHECK(branch->kind == VerkleNode::Kind::branch && branch->children[0x22] && branch->children[0x23],
          "shared-prefix stems split into an explicit intermediate branch node");
    CHECK(branch->children[0x22]->kind == VerkleNode::Kind::extension && branch->children[0x22]->suffixes.size() == 2,
          "same-stem keys share one explicit extension node with suffix slots");
}

void test_get_erase_and_persistence() {
    const VerkleKey a = key(0x11, 0x22, 0x01);
    const VerkleKey b = key(0x11, 0x22, 0x02);
    const VerkleKey c = key(0xee, 0xff, 0x03);
    Eip6800StateTree original;
    original.set(a, value(1)); original.set(b, VerkleValue{}); original.set(c, value(3));
    VerkleValue read{};
    CHECK(original.get(b, read) && read == VerkleValue{}, "get returns a present all-zero suffix value");
    const PointExtended expected_root = original.root();
    const std::vector<uint8_t> snapshot = original.serialize();
    Eip6800StateTree restored;
    CHECK(Eip6800StateTree::deserialize(snapshot, restored) && restored.size() == 3 && same(expected_root, restored.root()),
          "canonical persisted state reloads with identical root and key count");
    const std::string path = "/tmp/cuda_verkle_state_tree_test.vkl";
    Eip6800StateTree file_restored;
    CHECK(original.save(path) && Eip6800StateTree::load(path, file_restored) && same(expected_root, file_restored.root()),
          "state tree saves and loads canonical persistent snapshots");
    std::remove(path.c_str());
    CHECK(restored.erase(b) && !restored.contains(b) && restored.contains(a) && restored.contains(c),
          "erase removes one suffix while preserving sibling extension and branch nodes");
    const PointExtended root_before_failed_load = restored.root();

    std::vector<uint8_t> malformed = snapshot;
    malformed[0] = 'X';
    CHECK(!Eip6800StateTree::deserialize(malformed, restored), "persistence loader rejects invalid magic");
    CHECK(same(root_before_failed_load, restored.root()), "failed persistence load leaves existing tree unchanged");
    malformed = snapshot;
    if (malformed.size() >= 16 + 128) std::swap(malformed[16 + 31], malformed[16 + 64 + 31]);
    CHECK(!Eip6800StateTree::deserialize(malformed, restored), "persistence loader rejects non-canonical record ordering");
}

void test_key_derivation_and_bounds() {
    uint8_t address[32] = {}, tree_index[32] = {};
    address[31] = 0x42; tree_index[0] = 64;
    VerkleKey a, b;
    CHECK(Eip6800StateTree::state_key(address, tree_index, 3, a), "EIP-6800 state key derivation succeeds");
    CHECK(Eip6800StateTree::state_key(address, tree_index, 4, b) && std::memcmp(a.data(), b.data(), 31) == 0 && a[31] != b[31],
          "state key preserves the requested suffix after Pedersen stem hash");
    uint8_t output[32], oversized[4081] = {};
    CHECK(!Eip6800StateTree::pedersen_hash(oversized, sizeof(oversized), output), "Pedersen hash rejects inputs above EIP-6800 limit");
}

void test_rust_group_mapping_vectors() {
    using namespace cuda_verkle::test_util;
    const JsonValue vectors = read_json(vector_path("group_to_scalar_test_vectors.json"));
    bool matches = true;
    for (const JsonValue& item : vectors.at("test_cases").array) {
        uint8_t encoded[32]; uint32_t raw[8]; BanderwagonElement point;
        hex32(item.at("point").as_string(), encoded);
        load_hex_to_limbs(item.at("scalar").as_string(), raw);
        matches = matches && bw_from_bytes_strict(encoded, point) && fr_eq(bw_map_to_scalar_field(point), fr_from_raw(raw));
    }
    CHECK(matches, "EIP-6800 group-to-scalar mapping matches pinned rust-verkle vectors");
}

int main() {
    std::printf("EIP-6800 State Tree Tests\n");
    test_empty_and_zero_value(); test_determinism_and_branching(); test_get_erase_and_persistence(); test_key_derivation_and_bounds(); test_rust_group_mapping_vectors();
    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}
