// EIP-6800 sparse Verkle state tree (host-side reference implementation).
#pragma once

#include "../constants/crs_points.cuh"
#include "../curve/banderwagon.cuh"
#include "../msm/msm_kernel.cuh"
#include <array>
#include <cstdint>
#include <map>
#include <vector>

static constexpr int VERKLE_NODE_WIDTH = 256;
using VerkleKey = std::array<uint8_t, 32>;
using VerkleValue = std::array<uint8_t, 32>;
using VerkleStem = std::array<uint8_t, 31>;

// Values are present even when all their bytes are zero; erase() alone makes a key absent. The root follows EIP-6800's extension/suffix and main-tree rules.
class Eip6800StateTree {
public:
    Eip6800StateTree() { crs::load_crs(crs_); }
    void set(const VerkleKey& key, const VerkleValue& value) { values_[key] = value; }
    void erase(const VerkleKey& key) { values_.erase(key); }
    bool contains(const VerkleKey& key) const { return values_.find(key) != values_.end(); }
    size_t size() const { return values_.size(); }

    PointExtended root() const {
        if (values_.empty()) return point_identity();
        std::map<VerkleStem, std::vector<std::pair<uint8_t, VerkleValue>>> grouped;
        for (const auto& item : values_) {
            VerkleStem stem{};
            for (int i = 0; i < 31; ++i) stem[i] = item.first[i];
            grouped[stem].push_back({item.first[31], item.second});
        }
        std::vector<Entry> entries;
        for (const auto& item : grouped) entries.push_back({item.first, extension_scalar(item.first, item.second)});
        Fr children[VERKLE_NODE_WIDTH];
        for (int i = 0; i < VERKLE_NODE_WIDTH; ++i) children[i] = FR_ZERO;
        for (int byte = 0; byte < VERKLE_NODE_WIDTH; ++byte) {
            std::vector<Entry> child;
            for (const Entry& entry : entries) if (entry.stem[0] == byte) child.push_back(entry);
            children[byte] = main_scalar(child, 1);
        }
        return commit(children);
    }

    void root_bytes(uint8_t out[32]) const { bw_to_bytes({root()}, out); }

    // EIP-6800 pedersen_hash. Output is the little-endian Fr encoding of the
    // group_to_scalar_field result. Input is limited by the protocol to 4080 bytes.
    static bool pedersen_hash(const uint8_t* input, size_t length, uint8_t out[32]) {
        if (length > 255U * 16U) return false;
        Fr scalars[VERKLE_NODE_WIDTH];
        for (int i = 0; i < VERKLE_NODE_WIDTH; ++i) scalars[i] = FR_ZERO;
        scalars[0] = fr_from_u64(2 + 256ULL * length);
        for (size_t chunk = 0; chunk < 255; ++chunk) {
            uint32_t raw[8] = {};
            for (size_t j = 0; j < 16 && chunk * 16 + j < length; ++j)
                raw[j / 4] |= static_cast<uint32_t>(input[chunk * 16 + j]) << (8 * (j % 4));
            scalars[chunk + 1] = fr_from_raw(raw);
        }
        crs::CRSPoints crs_points;
        crs::load_crs(crs_points);
        Fr mapped = bw_map_to_scalar_field({msm_compute(scalars, crs_points.x, crs_points.y, VERKLE_NODE_WIDTH)});
        uint8_t big_endian[32];
        fr_to_bytes(mapped, big_endian);
        for (int i = 0; i < 32; ++i) out[i] = big_endian[31 - i];
        return true;
    }

    // tree_index_le is the protocol's 32-byte little-endian tree index.
    static bool state_key(const uint8_t address[32], const uint8_t tree_index_le[32], uint8_t sub_index, VerkleKey& out) {
        uint8_t input[64];
        for (int i = 0; i < 32; ++i) { input[i] = address[i]; input[32 + i] = tree_index_le[i]; }
        uint8_t hash[32];
        if (!pedersen_hash(input, sizeof(input), hash)) return false;
        for (int i = 0; i < 31; ++i) out[i] = hash[i];
        out[31] = sub_index;
        return true;
    }

private:
    struct Entry { VerkleStem stem; Fr value; };
    crs::CRSPoints crs_;
    std::map<VerkleKey, VerkleValue> values_;

    PointExtended commit(const Fr values[VERKLE_NODE_WIDTH]) const { return msm_compute(values, crs_.x, crs_.y, VERKLE_NODE_WIDTH); }
    static Fr little_endian_scalar(const uint8_t* bytes, size_t length, bool marker) {
        uint32_t raw[8] = {};
        for (size_t i = 0; i < length; ++i) raw[i / 4] |= static_cast<uint32_t>(bytes[i]) << (8 * (i % 4));
        if (marker) raw[4] |= 1U; // +2^128 distinguishes present zero from absent.
        return fr_from_raw(raw);
    }
    Fr extension_scalar(const VerkleStem& stem, const std::vector<std::pair<uint8_t, VerkleValue>>& leaves) const {
        Fr leaves_as_scalars[512];
        for (int i = 0; i < 512; ++i) leaves_as_scalars[i] = FR_ZERO;
        for (const auto& leaf : leaves) {
            leaves_as_scalars[2 * leaf.first] = little_endian_scalar(leaf.second.data(), 16, true);
            leaves_as_scalars[2 * leaf.first + 1] = little_endian_scalar(leaf.second.data() + 16, 16, false);
        }
        Fr extension[VERKLE_NODE_WIDTH];
        for (int i = 0; i < VERKLE_NODE_WIDTH; ++i) extension[i] = FR_ZERO;
        extension[0] = FR_ONE;
        extension[1] = little_endian_scalar(stem.data(), stem.size(), false);
        extension[2] = bw_map_to_scalar_field({commit(leaves_as_scalars)});
        extension[3] = bw_map_to_scalar_field({commit(leaves_as_scalars + VERKLE_NODE_WIDTH)});
        return bw_map_to_scalar_field({commit(extension)});
    }
    Fr main_scalar(const std::vector<Entry>& entries, int depth) const {
        if (entries.empty()) return FR_ZERO;
        if (entries.size() == 1) return entries[0].value;
        Fr children[VERKLE_NODE_WIDTH];
        for (int i = 0; i < VERKLE_NODE_WIDTH; ++i) children[i] = FR_ZERO;
        for (int byte = 0; byte < VERKLE_NODE_WIDTH; ++byte) {
            std::vector<Entry> child;
            for (const Entry& entry : entries) if (depth < 31 && entry.stem[depth] == byte) child.push_back(entry);
            children[byte] = main_scalar(child, depth + 1);
        }
        return bw_map_to_scalar_field({commit(children)});
    }
};
