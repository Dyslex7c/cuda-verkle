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

// Owns batch-specific device allocations. Create one workspace per CUDA stream
// when operations need to overlap. The members are opaque so including this
// header does not require CUDA headers in host-only builds.
struct MsmGpuWorkspace {
    void* device_scalars = nullptr;
    void* device_scalar_raw = nullptr;
    void* device_window_sums = nullptr;
    void* device_result = nullptr;
    int batch_capacity = 0;
    bool initialized = false;

    MsmGpuWorkspace() = default;
    MsmGpuWorkspace(const MsmGpuWorkspace&) = delete;
    MsmGpuWorkspace& operator=(const MsmGpuWorkspace&) = delete;
    MsmGpuWorkspace(MsmGpuWorkspace&&) = delete;
    MsmGpuWorkspace& operator=(MsmGpuWorkspace&&) = delete;
#ifdef __CUDACC__
    // Best-effort cleanup for every exit path. Call msm_gpu_workspace_destroy
    // explicitly when the caller needs to observe a CUDA cleanup error.
    ~MsmGpuWorkspace();
#else
    ~MsmGpuWorkspace() = default;
#endif
};

// Owns the GPU-resident fixed CRS plus a one-element default workspace for the
// synchronous single-MSM API. Contexts are not thread-safe; callers that use
// multiple streams should share no workspace between in-flight operations.
struct MsmGpuContext {
    void* device_point_x = nullptr;
    void* device_point_y = nullptr;
    MsmGpuWorkspace default_workspace;
    bool initialized = false;

    MsmGpuContext() = default;
    MsmGpuContext(const MsmGpuContext&) = delete;
    MsmGpuContext& operator=(const MsmGpuContext&) = delete;
    MsmGpuContext(MsmGpuContext&&) = delete;
    MsmGpuContext& operator=(MsmGpuContext&&) = delete;
#ifdef __CUDACC__
    // Best-effort cleanup for the fixed CRS and default workspace. Call
    // msm_gpu_context_destroy explicitly when cleanup status matters.
    ~MsmGpuContext();
#else
    ~MsmGpuContext() = default;
#endif
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

// Allocates batch-specific input, scalar-window, and output storage. The CRS
// remains owned by MsmGpuContext and is shared by all workspaces.
MsmGpuStatus msm_gpu_workspace_init(MsmGpuWorkspace& workspace, int batch_capacity);
MsmGpuStatus msm_gpu_workspace_destroy(MsmGpuWorkspace& workspace);

// Executes one fixed-width MSM using a windowed GPU Pippenger pipeline. It
// converts Montgomery scalars once, builds one bucket table per window in
// parallel, and combines the resulting window sums deterministically.
MsmGpuStatus msm_gpu_compute(
    MsmGpuContext& context,
    const Fr scalars[],
    int n,
    PointExtended* result);

// Enqueues a batch of fixed-width (256-scalar) GPU Pippenger MSMs. Inputs and
// outputs are contiguous arrays of batch_count * MSM_SIZE scalars and
// batch_count points respectively. The host buffers must remain valid until
// msm_gpu_stream_synchronize(stream) completes. A workspace cannot be reused
// until its previously enqueued operation on that stream has completed.
MsmGpuStatus msm_gpu_compute_batch_async(
    MsmGpuContext& context,
    MsmGpuWorkspace& workspace,
    const Fr scalars[],
    int batch_count,
    PointExtended results[],
    cudaStream_t stream = 0);

// Synchronous convenience form of msm_gpu_compute_batch_async on the default
// stream. It is useful for batching without managing stream lifetimes.
MsmGpuStatus msm_gpu_compute_batch(
    MsmGpuContext& context,
    MsmGpuWorkspace& workspace,
    const Fr scalars[],
    int batch_count,
    PointExtended results[]);

// Waits for enqueued copies and kernels in `stream`, making async results safe
// to read and allowing the associated workspace to be reused.
MsmGpuStatus msm_gpu_stream_synchronize(cudaStream_t stream = 0);

// Releases device allocations.  It is safe to call after a failed init or more
// than once; callers must invoke it before destroying a successfully
// initialized context.
MsmGpuStatus msm_gpu_context_destroy(MsmGpuContext& context);
#endif
