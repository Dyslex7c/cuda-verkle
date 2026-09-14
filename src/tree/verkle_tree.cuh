// EIP-6800 sparse Verkle state tree, host-side reference implementation.
#pragma once

#include "../constants/crs_points.cuh"
#include "../curve/banderwagon.cuh"
#include "../msm/msm_kernel.cuh"
#include <array>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

static constexpr int VERKLE_NODE_WIDTH = 256;
using VerkleKey = std::array<uint8_t, 32>;
using VerkleValue = std::array<uint8_t, 32>;
using VerkleStem = std::array<uint8_t, 31>;

// A branch is a 256-way main-tree node. An extension is a complete 31-byte
// stem and its 256 suffix/value slots. A value of all zero bytes is present;
// only erase() makes a suffix absent.
struct VerkleNode {
    enum class Kind : uint8_t { branch, extension };
    explicit VerkleNode(Kind node_kind) : kind(node_kind) {}
    Kind kind;
    std::array<std::unique_ptr<VerkleNode>, VERKLE_NODE_WIDTH> children{};
    VerkleStem stem{};
    std::map<uint8_t, VerkleValue> suffixes;
};

class Eip6800StateTree {
public:
    Eip6800StateTree() { crs::load_crs(crs_); root_ = make_branch(); }
    Eip6800StateTree(const Eip6800StateTree&) = delete;
    Eip6800StateTree& operator=(const Eip6800StateTree&) = delete;
    Eip6800StateTree(Eip6800StateTree&&) = default;
    Eip6800StateTree& operator=(Eip6800StateTree&&) = default;

    const VerkleNode& root_node() const { return *root_; }
    size_t size() const { return size_; }

    bool get(const VerkleKey& key, VerkleValue& out) const {
        const VerkleNode* node = root_.get();
        for (int depth = 0; node && node->kind == VerkleNode::Kind::branch; ++depth) {
            if (depth >= 31) return false;
            node = node->children[key[depth]].get();
        }
        if (!node || node->kind != VerkleNode::Kind::extension || !same_stem(node->stem, key)) return false;
        const auto value = node->suffixes.find(key[31]);
        if (value == node->suffixes.end()) return false;
        out = value->second;
        return true;
    }
    bool contains(const VerkleKey& key) const { VerkleValue ignored; return get(key, ignored); }

    void set(const VerkleKey& key, const VerkleValue& value) {
        insert(root_, 0, key, value);
    }

    bool erase(const VerkleKey& key) {
        if (!erase_from(root_, 0, key)) return false;
        // The root is permanently a branch because the EIP root is always a
        // 256-way commitment, including a tree with one stem.
        if (!root_) root_ = make_branch();
        return true;
    }

    PointExtended root() const { return branch_commitment(*root_, 0, true); }
    void root_bytes(uint8_t out[32]) const { bw_to_bytes({root()}, out); }

    // Canonical persistence format: magic "VKL1", format version 1, count
    // (u64 big-endian), then lexicographically ordered 32-byte key/value
    // records. Nodes are reconstructed from these canonical records on load.
    // This intentionally persists logical state rather than mutable cache or
    // node topology, preventing stale commitments after a reload.
    std::vector<uint8_t> serialize() const {
        std::vector<std::pair<VerkleKey, VerkleValue>> records;
        collect(*root_, records);
        std::vector<uint8_t> out;
        out.reserve(16 + records.size() * 64);
        out.insert(out.end(), {'V', 'K', 'L', '1'});
        append_u32(out, 1);
        append_u64(out, records.size());
        for (const auto& record : records) {
            out.insert(out.end(), record.first.begin(), record.first.end());
            out.insert(out.end(), record.second.begin(), record.second.end());
        }
        return out;
    }

    static bool deserialize(const std::vector<uint8_t>& bytes, Eip6800StateTree& out) {
        if (bytes.size() < 16 || bytes[0] != 'V' || bytes[1] != 'K' || bytes[2] != 'L' || bytes[3] != '1' ||
            read_u32(bytes, 4) != 1) return false;
        const uint64_t count = read_u64(bytes, 8);
        if (count > (bytes.size() - 16) / 64 || bytes.size() != 16 + count * 64) return false;
        Eip6800StateTree candidate;
        VerkleKey previous{};
        bool have_previous = false;
        for (uint64_t i = 0; i < count; ++i) {
            const size_t offset = 16 + static_cast<size_t>(i) * 64;
            VerkleKey key{}; VerkleValue value{};
            for (int j = 0; j < 32; ++j) { key[j] = bytes[offset + j]; value[j] = bytes[offset + 32 + j]; }
            if (have_previous && !(previous < key)) return false; // rejects duplicate and non-canonical order
            candidate.set(key, value);
            previous = key; have_previous = true;
        }
        out = std::move(candidate);
        return true;
    }

    // File helpers are thin wrappers over the canonical snapshot format above.
    // They never partially mutate an existing tree: load() validates a complete
    // file into a candidate tree before replacing `out`.
    bool save(const std::string& path) const {
        const std::vector<uint8_t> bytes = serialize();
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file) return false;
        file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        return file.good();
    }
    static bool load(const std::string& path, Eip6800StateTree& out) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) return false;
        const std::streamoff end = file.tellg();
        if (end < 0) return false;
        const size_t size = static_cast<size_t>(end);
        file.seekg(0, std::ios::beg);
        std::vector<uint8_t> bytes(size);
        if (size != 0) file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
        return file.good() && deserialize(bytes, out);
    }

    // EIP-6800 Pedersen hash: output is the little-endian Fr encoding of
    // group_to_scalar_field. The protocol limits input to 255 field chunks.
    static bool pedersen_hash(const uint8_t* input, size_t length, uint8_t out[32]) {
        if ((!input && length != 0) || length > 255U * 16U) return false;
        Fr scalars[VERKLE_NODE_WIDTH];
        for (int i = 0; i < VERKLE_NODE_WIDTH; ++i) scalars[i] = fr_zero();
        scalars[0] = fr_from_u64(2 + 256ULL * length);
        for (size_t chunk = 0; chunk < 255; ++chunk) {
            uint32_t raw[8] = {};
            for (size_t j = 0; j < 16 && chunk * 16 + j < length; ++j)
                raw[j / 4] |= static_cast<uint32_t>(input[chunk * 16 + j]) << (8 * (j % 4));
            scalars[chunk + 1] = fr_from_raw(raw);
        }
        crs::CRSPoints crs; crs::load_crs(crs);
        const Fr mapped = bw_map_to_scalar_field({msm_compute(scalars, crs.x, crs.y, VERKLE_NODE_WIDTH)});
        uint8_t big_endian[32]; fr_to_bytes(mapped, big_endian);
        for (int i = 0; i < 32; ++i) out[i] = big_endian[31 - i];
        return true;
    }

    static bool state_key(const uint8_t address[32], const uint8_t tree_index_le[32], uint8_t sub_index, VerkleKey& out) {
        if (!address || !tree_index_le) return false;
        uint8_t input[64], hash[32];
        for (int i = 0; i < 32; ++i) { input[i] = address[i]; input[32 + i] = tree_index_le[i]; }
        if (!pedersen_hash(input, sizeof(input), hash)) return false;
        for (int i = 0; i < 31; ++i) out[i] = hash[i];
        out[31] = sub_index;
        return true;
    }

private:
    crs::CRSPoints crs_;
    std::unique_ptr<VerkleNode> root_;
    size_t size_ = 0;

    static std::unique_ptr<VerkleNode> make_branch() { return std::make_unique<VerkleNode>(VerkleNode::Kind::branch); }
    static std::unique_ptr<VerkleNode> make_extension(const VerkleKey& key, const VerkleValue& value) {
        std::unique_ptr<VerkleNode> node = std::make_unique<VerkleNode>(VerkleNode::Kind::extension);
        for (int i = 0; i < 31; ++i) node->stem[i] = key[i];
        node->suffixes[key[31]] = value;
        return node;
    }
    static bool same_stem(const VerkleStem& stem, const VerkleKey& key) {
        for (int i = 0; i < 31; ++i) if (stem[i] != key[i]) return false;
        return true;
    }
    static int first_difference(const VerkleStem& stem, const VerkleKey& key, int from) {
        for (int i = from; i < 31; ++i) if (stem[i] != key[i]) return i;
        return 31;
    }

    void insert(std::unique_ptr<VerkleNode>& node, int depth, const VerkleKey& key, const VerkleValue& value) {
        if (!node) { node = make_extension(key, value); ++size_; return; }
        if (node->kind == VerkleNode::Kind::branch) { insert(node->children[key[depth]], depth + 1, key, value); return; }
        if (same_stem(node->stem, key)) {
            const bool was_present = node->suffixes.find(key[31]) != node->suffixes.end();
            node->suffixes[key[31]] = value;
            if (!was_present) ++size_;
            return;
        }
        const int split_at = first_difference(node->stem, key, depth);
        std::unique_ptr<VerkleNode> old = std::move(node);
        node = make_branch();
        VerkleNode* branch = node.get();
        for (int level = depth; level < split_at; ++level) {
            branch->children[old->stem[level]] = make_branch();
            branch = branch->children[old->stem[level]].get();
        }
        branch->children[old->stem[split_at]] = std::move(old);
        branch->children[key[split_at]] = make_extension(key, value);
        ++size_;
    }

    bool erase_from(std::unique_ptr<VerkleNode>& node, int depth, const VerkleKey& key) {
        if (!node) return false;
        if (node->kind == VerkleNode::Kind::extension) {
            if (!same_stem(node->stem, key)) return false;
            const auto found = node->suffixes.find(key[31]);
            if (found == node->suffixes.end()) return false;
            node->suffixes.erase(found); --size_;
            if (node->suffixes.empty()) node.reset();
            return true;
        }
        if (depth >= 31) return false;
        const uint8_t index = key[depth];
        if (!erase_from(node->children[index], depth + 1, key)) return false;
        collapse_branch(node, depth);
        return true;
    }

    static void collapse_branch(std::unique_ptr<VerkleNode>& node, int depth) {
        if (!node || node->kind != VerkleNode::Kind::branch || depth == 0) return;
        int count = 0; VerkleNode* only = nullptr;
        for (const auto& child : node->children) if (child) { ++count; only = child.get(); }
        if (count == 0) { node.reset(); return; }
        if (count == 1 && only->kind == VerkleNode::Kind::extension) {
            for (auto& child : node->children) if (child) { node = std::move(child); return; }
        }
    }

    PointExtended commit(const Fr values[VERKLE_NODE_WIDTH]) const { return msm_compute(values, crs_.x, crs_.y, VERKLE_NODE_WIDTH); }
    static Fr little_endian_scalar(const uint8_t* bytes, size_t length, bool marker) {
        uint32_t raw[8] = {};
        for (size_t i = 0; i < length; ++i) raw[i / 4] |= static_cast<uint32_t>(bytes[i]) << (8 * (i % 4));
        if (marker) raw[4] |= 1U;
        return fr_from_raw(raw);
    }
    PointExtended extension_commitment(const VerkleNode& node) const {
        Fr suffix_scalars[512];
        for (int i = 0; i < 512; ++i) suffix_scalars[i] = fr_zero();
        for (const auto& item : node.suffixes) {
            suffix_scalars[2 * item.first] = little_endian_scalar(item.second.data(), 16, true);
            suffix_scalars[2 * item.first + 1] = little_endian_scalar(item.second.data() + 16, 16, false);
        }
        Fr extension[VERKLE_NODE_WIDTH];
        for (int i = 0; i < VERKLE_NODE_WIDTH; ++i) extension[i] = fr_zero();
        extension[0] = fr_one();
        extension[1] = little_endian_scalar(node.stem.data(), node.stem.size(), false);
        extension[2] = bw_map_to_scalar_field({commit(suffix_scalars)});
        extension[3] = bw_map_to_scalar_field({commit(suffix_scalars + VERKLE_NODE_WIDTH)});
        return commit(extension);
    }
    Fr node_scalar(const VerkleNode& node, int depth) const {
        if (node.kind == VerkleNode::Kind::extension) return bw_map_to_scalar_field({extension_commitment(node)});
        return bw_map_to_scalar_field({branch_commitment(node, depth, false)});
    }
    PointExtended branch_commitment(const VerkleNode& node, int depth, bool force_commitment) const {
        int count = 0; const VerkleNode* only = nullptr;
        for (const auto& child : node.children) if (child) { ++count; only = child.get(); }
        if (!force_commitment && count == 0) return point_identity();
        if (!force_commitment && count == 1 && only->kind == VerkleNode::Kind::extension) return extension_commitment(*only);
        Fr children[VERKLE_NODE_WIDTH];
        for (int i = 0; i < VERKLE_NODE_WIDTH; ++i)
            children[i] = node.children[i] ? node_scalar(*node.children[i], depth + 1) : fr_zero();
        return commit(children);
    }
    static void collect(const VerkleNode& node, std::vector<std::pair<VerkleKey, VerkleValue>>& out) {
        if (node.kind == VerkleNode::Kind::extension) {
            for (const auto& item : node.suffixes) {
                VerkleKey key{}; for (int i = 0; i < 31; ++i) key[i] = node.stem[i]; key[31] = item.first;
                out.push_back({key, item.second});
            }
            return;
        }
        for (const auto& child : node.children) if (child) collect(*child, out);
    }
    static void append_u32(std::vector<uint8_t>& out, uint32_t value) { for (int i = 3; i >= 0; --i) out.push_back(static_cast<uint8_t>(value >> (8 * i))); }
    static void append_u64(std::vector<uint8_t>& out, uint64_t value) { for (int i = 7; i >= 0; --i) out.push_back(static_cast<uint8_t>(value >> (8 * i))); }
    static uint32_t read_u32(const std::vector<uint8_t>& in, size_t offset) { uint32_t r = 0; for (int i = 0; i < 4; ++i) r = (r << 8) | in[offset + i]; return r; }
    static uint64_t read_u64(const std::vector<uint8_t>& in, size_t offset) { uint64_t r = 0; for (int i = 0; i < 8; ++i) r = (r << 8) | in[offset + i]; return r; }
};
