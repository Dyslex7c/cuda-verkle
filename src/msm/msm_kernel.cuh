#pragma once

#include "../field/fp.cuh"
#include "../field/fr.cuh"
#include "../curve/bandersnatch.cuh"
#include "../curve/banderwagon.cuh"

// Maximum number of points in a single MSM (fixed for Verkle tree nodes)
static constexpr int MSM_SIZE = 256;
static constexpr int MSM_WINDOW_BITS = 8;
static constexpr int MSM_BUCKET_COUNT = 1 << MSM_WINDOW_BITS;
static constexpr int MSM_NUM_WINDOWS = (253 + MSM_WINDOW_BITS - 1) / MSM_WINDOW_BITS;

// Status returned by the CUDA MSM API.  The CUDA path is intentionally
// separate from the CPU Pippenger reference below: it owns device memory and
// launches a real kernel when compiled with nvcc.
enum class MsmGpuStatus {
    success = 0,
    invalid_argument,
    no_cuda_device,
    cuda_error,
    not_initialized,
};

// Owns GPU allocations for one fixed 256-point CRS and one MSM input/output.
// The members are opaque so including this header does not require CUDA headers
// in host-only builds.  A context is not thread-safe; use one per host thread.
struct MsmGpuContext {
    void* device_scalars = nullptr;
    void* device_point_x = nullptr;
    void* device_point_y = nullptr;
    void* device_scalar_raw = nullptr;
    void* device_window_sums = nullptr;
    void* device_result = nullptr;
    bool initialized = false;

    MsmGpuContext() = default;
    MsmGpuContext(const MsmGpuContext&) = delete;
    MsmGpuContext& operator=(const MsmGpuContext&) = delete;
    MsmGpuContext(MsmGpuContext&&) = delete;
    MsmGpuContext& operator=(MsmGpuContext&&) = delete;
};

// Compute a multi-scalar multiplication result = sum(scalars[i] * points[i]) This is the CPU reference implementation used for correctness testing.
__host__ __device__ PointExtended msm_cpu_reference(
    const Fr scalars[],
    const Fp point_x[],  // SoA: x-coordinates of basis points (affine, Montgomery form)
    const Fp point_y[],  // SoA: y-coordinates
    int n);

// Compute one MSM using the sequential CPU Pippenger implementation. This
// host/device-compatible function is the reference used by host tests; the
// launched CUDA Pippenger implementation and its device-memory API are
// declared below.
__host__ __device__ PointExtended msm_compute(
    const Fr scalars[],
    const Fp point_x[],
    const Fp point_y[],
    int n);

#ifdef __CUDACC__
// Uploads the fixed CRS once.  `point_x` and `point_y` must each contain
// MSM_SIZE affine Montgomery-form coordinates.
MsmGpuStatus msm_gpu_context_init(
    MsmGpuContext& context,
    const Fp point_x[MSM_SIZE],
    const Fp point_y[MSM_SIZE]);

// Executes one fixed-width MSM using a windowed GPU Pippenger pipeline. It
// converts Montgomery scalars once, builds one bucket table per window in
// parallel, and combines the resulting window sums deterministically.
MsmGpuStatus msm_gpu_compute(
    MsmGpuContext& context,
    const Fr scalars[],
    int n,
    PointExtended* result);

// Releases device allocations.  It is safe to call after a failed init or more
// than once; callers must invoke it before destroying a successfully
// initialized context.
MsmGpuStatus msm_gpu_context_destroy(MsmGpuContext& context);
#endif
