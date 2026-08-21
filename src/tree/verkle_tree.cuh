// Implements a 1-2 level deep 256-ary commitment tree where each node's commitment is a Pedersen vector commitment over its children's values/commitments.
// See details: https://eips.ethereum.org/EIPS/eip-6800. This implementation below is a simulation.

#pragma once

#include "../field/fp.cuh"
#include "../field/fr.cuh"
#include "../curve/bandersnatch.cuh"
#include "../curve/banderwagon.cuh"
#include "../constants/crs_points.cuh"
#include "../msm/msm_kernel.cuh"
#include <cstring>
#include <cstdio>

// max tree depth for simulation
static constexpr int MAX_TREE_DEPTH = 3;
static constexpr int TREE_WIDTH = 256;

// leaf update: (index, new_value)
struct LeafUpdate {
    int index; // leaf index
    Fr new_value; // new scalar value for this leaf
};

// node commitment result (a Banderwagon element)
struct NodeCommitment {
    PointExtended point;
    bool computed;
};

// simulated Verkle tree for benchmarking and testing.
//
// For a 1-level tree: 256 leaves → 1 root commitment
// For a 2-level tree: 256*256 = 65,536 leaves → 256 L1 nodes → 1 root
//
// Each node's commitment = MSM(child_values, CRS_basis)
struct VerkleTree {
    int depth; // 1 or 2 (number of levels of internal nodes)
    crs::CRSPoints crs; // Preloaded CRS basis points
    
    // Level 0 (root): single commitment over 256 level-1 commitments
    // Level 1: 256 nodes, each committing 256 leaf values
    // Leaves: depth * 256 values at the bottom level
    
    // For depth=1: leaves[0..255], root = MSM(leaves, CRS)
    // For depth=2: leaves[0..65535], l1[i] = MSM(leaves[i*256..(i+1)*256-1], CRS),
    //              root = MSM(l1_as_scalars, CRS)
    
    // leaf storage
    Fr* leaves = nullptr;
    int num_leaves = 0;
    
    NodeCommitment l1_commitments[TREE_WIDTH];
    
    NodeCommitment root;
    
    void init(int d) {
        cleanup();
        depth = (d < 1) ? 1 : ((d > 2) ? 2 : d);
        crs::load_crs(crs);
        
        if (depth == 1) {
            num_leaves = TREE_WIDTH;
        } else {
            num_leaves = TREE_WIDTH * TREE_WIDTH;
        }
        
        leaves = new Fr[num_leaves];
        for (int i = 0; i < num_leaves; ++i) {
            leaves[i] = FR_ZERO;
        }
        
        for (int i = 0; i < TREE_WIDTH; ++i) {
            l1_commitments[i].point = point_identity();
            l1_commitments[i].computed = false;
        }
        root.point = point_identity();
        root.computed = false;
    }
    
    void cleanup() {
        delete[] leaves;
        leaves = nullptr;
        num_leaves = 0;
    }
    
    // set all leaf values and recompute the full tree
    void set_leaves(const Fr* values, int count) {
        int n = (count < 0) ? 0 : ((count < num_leaves) ? count : num_leaves);
        for (int i = 0; i < n; ++i) {
            leaves[i] = values[i];
        }
        // `set_leaves` replaces the complete logical leaf set. Clear an old suffix when a shorter input is supplied instead of retaining state.
        for (int i = n; i < num_leaves; ++i) {
            leaves[i] = FR_ZERO;
        }
        recompute_full();
    }
    
    // recompute all commitments from scratch
    void recompute_full() {
        if (depth == 1) {
            // Root = MSM(leaves[0..255], CRS)
            root.point = msm_compute(leaves, crs.x, crs.y, TREE_WIDTH);
            root.computed = true;
        } else {
            // compute each L1 node
            for (int i = 0; i < TREE_WIDTH; ++i) {
                l1_commitments[i].point = msm_compute(
                    &leaves[i * TREE_WIDTH], crs.x, crs.y, TREE_WIDTH);
                l1_commitments[i].computed = true;
            }
            
            // we use the commitment's serialized x-coordinate mapped to Fr for now for simplicity
            Fr l1_scalars[TREE_WIDTH];
            for (int i = 0; i < TREE_WIDTH; ++i) {
                commitment_to_scalar(l1_commitments[i].point, l1_scalars[i]);
            }
            root.point = msm_compute(l1_scalars, crs.x, crs.y, TREE_WIDTH);
            root.computed = true;
        }
    }
    
    void apply_updates_incremental(const LeafUpdate* updates, int num_updates) {
        if (depth == 1) {
            // delta update on root: for each update: root += (new_val - old_val) * G_i
            for (int u = 0; u < num_updates; ++u) {
                int idx = updates[u].index;
                if (idx < 0 || idx >= num_leaves) continue;
                
                Fr old_val = leaves[idx];
                Fr new_val = updates[u].new_value;
                Fr delta = fr_sub(new_val, old_val);
                
                if (!fr_is_zero(delta)) {
                    // compute delta * G_idx
                    PointAffine g_aff = {crs.x[idx], crs.y[idx]};
                    PointExtended g_ext = point_from_affine(g_aff);
                    PointExtended delta_point = scalar_mul(g_ext, delta);
                    root.point = point_add(root.point, delta_point);
                }
                
                leaves[idx] = new_val;
            }
        } else {
            // Track affected L1 nodes and their old root scalar values. This permits a delta update of the root rather than a full root MSM.
            bool l1_dirty[TREE_WIDTH] = {false};
            Fr old_l1_scalars[TREE_WIDTH];
            
            for (int u = 0; u < num_updates; ++u) {
                int idx = updates[u].index;
                if (idx < 0 || idx >= num_leaves) continue;
                
                int l1_idx = idx / TREE_WIDTH;
                int leaf_in_node = idx % TREE_WIDTH;
                if (!l1_dirty[l1_idx]) {
                    commitment_to_scalar(l1_commitments[l1_idx].point,
                                         old_l1_scalars[l1_idx]);
                    l1_dirty[l1_idx] = true;
                }
                Fr old_val = leaves[idx];
                Fr new_val = updates[u].new_value;
                Fr delta = fr_sub(new_val, old_val);
                
                if (!fr_is_zero(delta)) {
                    // delta update on L1 node
                    PointAffine g_aff = {crs.x[leaf_in_node], crs.y[leaf_in_node]};
                    PointExtended g_ext = point_from_affine(g_aff);
                    PointExtended delta_point = scalar_mul(g_ext, delta);
                    l1_commitments[l1_idx].point = point_add(
                        l1_commitments[l1_idx].point, delta_point);
                }
                
                leaves[idx] = new_val;
            }
            
            for (int i = 0; i < TREE_WIDTH; ++i) {
                if (!l1_dirty[i]) continue;
                Fr new_l1_scalar;
                commitment_to_scalar(l1_commitments[i].point, new_l1_scalar);
                Fr delta = fr_sub(new_l1_scalar, old_l1_scalars[i]);
                if (!fr_is_zero(delta)) {
                    PointAffine g_aff = {crs.x[i], crs.y[i]};
                    PointExtended g_ext = point_from_affine(g_aff);
                    root.point = point_add(root.point, scalar_mul(g_ext, delta));
                }
            }
            root.computed = true;
        }
    }
    
    void apply_updates_full(const LeafUpdate* updates, int num_updates) {
        for (int u = 0; u < num_updates; ++u) {
            int idx = updates[u].index;
            if (idx >= 0 && idx < num_leaves) {
                leaves[idx] = updates[u].new_value;
            }
        }
        recompute_full();
    }
    
    // get root commitment as serialized bytes
    void get_root_bytes(uint8_t out[32]) const {
        BanderwagonElement bw = {root.point};
        bw_to_bytes(bw, out);
    }
    
private:
    // convert a commitment point to a scalar field element. This maps the Banderwagon element to Fr by serializing and reducing mod n
    static void commitment_to_scalar(const PointExtended& pt, Fr& out) {
        BanderwagonElement bw = {pt};
        uint8_t bytes[32];
        bw_to_bytes(bw, bytes);
        
        // convert big-endian bytes to little-endian limbs and reduce mod n
        uint32_t limbs[8] = {0};
        for (int i = 0; i < 8; ++i) {
            int byte_offset = (7 - i) * 4;
            limbs[i] = ((uint32_t)bytes[byte_offset] << 24) |
                       ((uint32_t)bytes[byte_offset + 1] << 16) |
                       ((uint32_t)bytes[byte_offset + 2] << 8) |
                       ((uint32_t)bytes[byte_offset + 3]);
        }
        
        // fr_from_raw converts into Montgomery form and reduces the 256-bit serialized value modulo the Fr modulus.
        out = fr_from_raw(limbs);
    }
};
