#pragma once

#include "../field/fp.cuh"
#include "../field/fr.cuh"
#include "../curve/bandersnatch.cuh"
#include "../curve/banderwagon.cuh"

// Maximum number of points in a single MSM (fixed for Verkle tree nodes)
static constexpr int MSM_SIZE = 256;

// Compute a multi-scalar multiplication result = sum(scalars[i] * points[i]) This is the CPU reference implementation used for correctness testing.
__host__ __device__ PointExtended msm_cpu_reference(
    const Fr scalars[],
    const Fp point_x[],  // SoA: x-coordinates of basis points (affine, Montgomery form)
    const Fp point_y[],  // SoA: y-coordinates
    int n);

// compute one MSM using the sequential Pippenger implementation.
// This function is host/device compatible, but this repository does not yet provide a launched CUDA kernel or GPU memory-management API.
__host__ __device__ PointExtended msm_compute(
    const Fr scalars[],
    const Fp point_x[],
    const Fp point_y[],
    int n);
