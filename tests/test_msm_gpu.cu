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
    MsmGpuStatus destroy_status = msm_gpu_context_destroy(context);
    ASSERT_TRUE(destroy_status == MsmGpuStatus::success, "CUDA MSM context is released");

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
