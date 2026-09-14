#pragma once

#include "../field/fr.cuh"
#include "../curve/banderwagon.cuh"
#include "../constants/crs_points.cuh"
#include "../msm/msm_kernel.cuh"

// compute a Pedersen vector commitment over the Banderwagon group: C = sum(values[i] * G_i) for i in 0..n-1 where G_i are the CRS basis points.
struct PedersenCommitment {
    crs::CRSPoints crs;
    bool initialized = false;
    
    // Initialize by loading the CRS
    // CRS parsing is host-only. CUDA acceleration is provided through the
    // explicit MsmGpuContext API, which uploads this fixed CRS once.
    void init() {
        crs::load_crs(crs);
        initialized = true;
    }
    
    // Compute commitment for up to 256 scalar values
    BanderwagonElement commit(const Fr values[], int n) const;
    
    // Compute commitment and return serialized 32-byte form
    void commit_to_bytes(const Fr values[], int n, uint8_t out[32]) const;
};
