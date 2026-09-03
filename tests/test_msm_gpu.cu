// CUDA MSM integration tests. These compare the launched GPU kernel against
// the host Pippenger reference on the same CRS and scalar inputs.

#include <cstdio>
#include <cuda_runtime.h>

#include "../src/constants/crs_points.cuh"
#include "../src/msm/msm_kernel.cuh"
#include "../src/curve/banderwagon.cuh"

namespace {

int passed = 0;
int failed = 0;

#define ASSERT_TRUE(condition, message) do { \
    if (condition) { ++passed; std::printf("  PASS: %s\n", message); } \
    else { ++failed; std::printf("  FAIL: %s\n", message); } \
} while (0)

bool gpu_matches_cpu(
    MsmGpuContext& context,
    const Fr scalars[MSM_SIZE],
    int n,
    const crs::CRSPoints& crs_points) {
    PointExtended gpu_result;
    if (msm_gpu_compute(context, scalars, n, &gpu_result) != MsmGpuStatus::success) {
        return false;
    }
    PointExtended cpu_result = msm_compute(scalars, crs_points.x, crs_points.y, n);
    return bw_eq({gpu_result}, {cpu_result});
}

void fill_zero(Fr scalars[MSM_SIZE]) {
    for (int i = 0; i < MSM_SIZE; ++i) scalars[i] = FR_ZERO;
}

void test_gpu_msm(MsmGpuContext& context, const crs::CRSPoints& crs_points) {
    Fr scalars[MSM_SIZE];

    fill_zero(scalars);
    ASSERT_TRUE(gpu_matches_cpu(context, scalars, MSM_SIZE, crs_points),
                "GPU MSM matches CPU for zero scalars");

    fill_zero(scalars);
    scalars[17] = fr_from_u64(1);
    ASSERT_TRUE(gpu_matches_cpu(context, scalars, MSM_SIZE, crs_points),
                "GPU MSM matches CPU for one non-zero scalar");

    for (int i = 0; i < MSM_SIZE; ++i) scalars[i] = fr_from_u64(static_cast<uint64_t>(i + 1));
    ASSERT_TRUE(gpu_matches_cpu(context, scalars, MSM_SIZE, crs_points),
                "GPU MSM matches CPU for sequential scalars");

    // Deterministic non-trivial full-width input exercises all scalar bits
    // without requiring a host RNG dependency.
    uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < MSM_SIZE; ++i) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        scalars[i] = fr_from_u64(state * 0x2545f4914f6cdd1dULL);
    }
    ASSERT_TRUE(gpu_matches_cpu(context, scalars, MSM_SIZE, crs_points),
                "GPU MSM matches CPU for deterministic full-width input");

    PointExtended result;
    ASSERT_TRUE(msm_gpu_compute(context, scalars, 0, &result) == MsmGpuStatus::success &&
                    point_is_identity(result),
                "GPU MSM with zero points returns identity");
    ASSERT_TRUE(msm_gpu_compute(context, scalars, MSM_SIZE + 1, &result) ==
                    MsmGpuStatus::invalid_argument,
                "GPU MSM rejects point counts above the fixed CRS width");
}

void test_gpu_batch_msm(MsmGpuContext& context, const crs::CRSPoints& crs_points) {
    static constexpr int BATCH_SIZE = 3;
    Fr scalars[BATCH_SIZE][MSM_SIZE];
    PointExtended results[BATCH_SIZE];
    for (int batch = 0; batch < BATCH_SIZE; ++batch) {
        for (int i = 0; i < MSM_SIZE; ++i) {
            scalars[batch][i] = fr_from_u64(
                static_cast<uint64_t>(batch + 1) * (i + 3) * 101);
        }
    }

    MsmGpuWorkspace workspace;
    ASSERT_TRUE(msm_gpu_workspace_init(workspace, BATCH_SIZE) == MsmGpuStatus::success,
                "GPU batch workspace initializes");
    if (!workspace.initialized) return;

    ASSERT_TRUE(msm_gpu_compute_batch(
                    context, workspace, &scalars[0][0], BATCH_SIZE, results) ==
                    MsmGpuStatus::success,
                "GPU batch MSM completes");
    for (int batch = 0; batch < BATCH_SIZE; ++batch) {
        PointExtended expected = msm_compute(
            scalars[batch], crs_points.x, crs_points.y, MSM_SIZE);
        ASSERT_TRUE(bw_eq({results[batch]}, {expected}),
                    "GPU batch result matches CPU Pippenger");
    }
    ASSERT_TRUE(msm_gpu_workspace_destroy(workspace) == MsmGpuStatus::success,
                "GPU batch workspace is released");
}

void test_gpu_streamed_msm(MsmGpuContext& context, const crs::CRSPoints& crs_points) {
    cudaStream_t streams[2] = {};
    MsmGpuWorkspace workspaces[2];
    Fr scalars[2][MSM_SIZE];
    PointExtended results[2];
    bool setup_ok = true;

    for (int stream_index = 0; stream_index < 2; ++stream_index) {
        if (cudaStreamCreateWithFlags(&streams[stream_index], cudaStreamNonBlocking) != cudaSuccess ||
            msm_gpu_workspace_init(workspaces[stream_index], 1) != MsmGpuStatus::success) {
            setup_ok = false;
            break;
        }
        for (int i = 0; i < MSM_SIZE; ++i) {
            scalars[stream_index][i] = fr_from_u64(
                static_cast<uint64_t>(stream_index + 7) * (i + 11) * 313);
        }
    }
    ASSERT_TRUE(setup_ok, "GPU stream workspaces initialize");
    if (setup_ok) {
        const MsmGpuStatus first = msm_gpu_compute_batch_async(
            context, workspaces[0], scalars[0], 1, &results[0], streams[0]);
        const MsmGpuStatus second = msm_gpu_compute_batch_async(
            context, workspaces[1], scalars[1], 1, &results[1], streams[1]);
        ASSERT_TRUE(first == MsmGpuStatus::success && second == MsmGpuStatus::success,
                    "GPU MSM batches enqueue on independent streams");
        ASSERT_TRUE(msm_gpu_stream_synchronize(streams[0]) == MsmGpuStatus::success &&
                        msm_gpu_stream_synchronize(streams[1]) == MsmGpuStatus::success,
                    "GPU MSM streams synchronize");
        for (int stream_index = 0; stream_index < 2; ++stream_index) {
            PointExtended expected = msm_compute(
                scalars[stream_index], crs_points.x, crs_points.y, MSM_SIZE);
            ASSERT_TRUE(bw_eq({results[stream_index]}, {expected}),
                        "GPU streamed result matches CPU Pippenger");
        }
    }

    for (int stream_index = 0; stream_index < 2; ++stream_index) {
        if (workspaces[stream_index].initialized) {
            ASSERT_TRUE(msm_gpu_workspace_destroy(workspaces[stream_index]) == MsmGpuStatus::success,
                        "GPU stream workspace is released");
        }
        if (streams[stream_index] != nullptr) cudaStreamDestroy(streams[stream_index]);
    }
}

} // namespace

int main() {
    std::printf("CUDA MSM Integration Tests\n");

    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no CUDA device is available\n");
        return 77;
    }

    crs::CRSPoints crs_points;
    crs::load_crs(crs_points);
    MsmGpuContext context;
    MsmGpuStatus init_status = msm_gpu_context_init(context, crs_points.x, crs_points.y);
    if (init_status != MsmGpuStatus::success) {
        std::printf("FAIL: unable to initialize CUDA MSM context (%d)\n",
                    static_cast<int>(init_status));
        return 1;
    }

    test_gpu_msm(context, crs_points);
    test_gpu_batch_msm(context, crs_points);
    test_gpu_streamed_msm(context, crs_points);
    MsmGpuStatus destroy_status = msm_gpu_context_destroy(context);
    ASSERT_TRUE(destroy_status == MsmGpuStatus::success, "CUDA MSM context is released");

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
