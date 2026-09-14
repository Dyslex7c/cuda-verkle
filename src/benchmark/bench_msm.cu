// bench_msm.cu - MSM and commitment benchmarks
// Measures timing for the core MSM operation at various scales.
// On CPU: uses <chrono> for timing.
// CPU baselines plus CUDA event-timed, batched Pippenger measurements.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include "../src/field/fp.cuh"
#include "../src/field/fr.cuh"
#include "../src/curve/bandersnatch.cuh"
#include "../src/curve/banderwagon.cuh"
#include "../src/constants/crs_points.cuh"
#include "../src/msm/msm_kernel.cuh"
#include "../src/util/cuda_raii.cuh"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

struct CpuTimer {
    std::chrono::high_resolution_clock::time_point start_;
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

void bench_msm_256_pippenger() {
    printf("\n[Benchmark] Pippenger MSM (256 points)\n");

    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    // generate test scalars (sequential for reproducibility)
    Fr scalars[256];
    for (int i = 0; i < 256; ++i) {
        scalars[i] = fr_from_u64((uint64_t)(i + 1) * 12345678901ULL);
    }

    // warmup
    PointExtended result = msm_compute(scalars, crs_pts.x, crs_pts.y, 256);
    (void)result;

    // benchmark
    int iterations = 10;
    CpuTimer timer;
    timer.start();
    for (int iter = 0; iter < iterations; ++iter) {
        result = msm_compute(scalars, crs_pts.x, crs_pts.y, 256);
    }
    double total_ms = timer.elapsed_ms();

    printf("  Pippenger MSM (256 points, w=8):\n");
    printf("    Total time for %d iterations: %.2f ms\n", iterations, total_ms);
    printf("    Average per MSM:              %.2f ms\n", total_ms / iterations);
    printf("    Throughput:                   %.0f MSM/s\n", iterations / (total_ms / 1000.0));
}

void bench_msm_256_naive() {
    printf("\n[Benchmark] Naive MSM (256 points)\n");

    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    Fr scalars[256];
    for (int i = 0; i < 256; ++i) {
        scalars[i] = fr_from_u64((uint64_t)(i + 1) * 12345678901ULL);
    }

    PointExtended result = msm_cpu_reference(scalars, crs_pts.x, crs_pts.y, 256);
    (void)result;

    // fewer iterations since naive is slow
    int iterations = 3;
    CpuTimer timer;
    timer.start();
    for (int iter = 0; iter < iterations; ++iter) {
        result = msm_cpu_reference(scalars, crs_pts.x, crs_pts.y, 256);
    }
    double total_ms = timer.elapsed_ms();

    printf("  Naive MSM (256 points):\n");
    printf("    Total time for %d iterations: %.2f ms\n", iterations, total_ms);
    printf("    Average per MSM:              %.2f ms\n", total_ms / iterations);
    printf("    Throughput:                   %.0f MSM/s\n", iterations / (total_ms / 1000.0));
}

void bench_msm_sparse() {
    printf("\n[Benchmark] Pippenger MSM (sparse, 10/256 non-zero)\n");

    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    Fr scalars[256];
    for (int i = 0; i < 256; ++i) scalars[i] = fr_zero();
    // only 10 non-zero entries
    for (int i = 0; i < 10; ++i) {
        scalars[i * 25] = fr_from_u64((uint64_t)(i + 1) * 999);
    }

    int iterations = 10;
    CpuTimer timer;
    timer.start();
    for (int iter = 0; iter < iterations; ++iter) {
        PointExtended result = msm_compute(scalars, crs_pts.x, crs_pts.y, 256);
        (void)result;
    }
    double total_ms = timer.elapsed_ms();

    printf("  Sparse MSM (10/256 non-zero):\n");
    printf("    Average per MSM: %.2f ms\n", total_ms / iterations);
}

void bench_field_ops() {
    printf("\n[Benchmark] Field Operations (Fp)\n");

    Fp a = fp_from_u64(123456789);
    Fp b = fp_from_u64(987654321);

    int iterations = 1000000;
    CpuTimer timer;

    // fp multiplication
    timer.start();
    Fp c = a;
    for (int i = 0; i < iterations; ++i) {
        c = fp_mul(c, b);
    }
    double mul_ms = timer.elapsed_ms();
    printf("  fp_mul:  %d ops in %.2f ms  (%.0f ns/op, %.1f M ops/s)\n",
           iterations, mul_ms, mul_ms * 1e6 / iterations, iterations / mul_ms / 1000.0);

    // fp addition
    timer.start();
    c = a;
    for (int i = 0; i < iterations; ++i) {
        c = fp_add(c, b);
    }
    double add_ms = timer.elapsed_ms();
    printf("  fp_add:  %d ops in %.2f ms  (%.0f ns/op, %.1f M ops/s)\n",
           iterations, add_ms, add_ms * 1e6 / iterations, iterations / add_ms / 1000.0);

    // fp squaring
    timer.start();
    c = a;
    for (int i = 0; i < iterations; ++i) {
        c = fp_sqr(c);
    }
    double sqr_ms = timer.elapsed_ms();
    printf("  fp_sqr:  %d ops in %.2f ms  (%.0f ns/op, %.1f M ops/s)\n",
           iterations, sqr_ms, sqr_ms * 1e6 / iterations, iterations / sqr_ms / 1000.0);

    // prevent dead code elimination
    if (fp_is_zero(c)) printf("(prevent DCE)\n");
}

void bench_point_ops() {
    printf("\n[Benchmark] Point Operations\n");

    crs::CRSPoints crs_pts;
    crs::load_crs(crs_pts);

    PointAffine g0_aff = {crs_pts.x[0], crs_pts.y[0]};
    PointExtended G0 = point_from_affine(g0_aff);
    PointAffine g1_aff = {crs_pts.x[1], crs_pts.y[1]};
    PointExtended G1 = point_from_affine(g1_aff);

    int iterations = 100000;
    CpuTimer timer;

    // point addition
    timer.start();
    PointExtended p = G0;
    for (int i = 0; i < iterations; ++i) {
        p = point_add(p, G1);
    }
    double add_ms = timer.elapsed_ms();
    printf("  point_add:    %d ops in %.2f ms  (%.1f us/op)\n",
           iterations, add_ms, add_ms * 1000.0 / iterations);

    // point doubling
    timer.start();
    p = G0;
    for (int i = 0; i < iterations; ++i) {
        p = point_double(p);
    }
    double dbl_ms = timer.elapsed_ms();
    printf("  point_double: %d ops in %.2f ms  (%.1f us/op)\n",
           iterations, dbl_ms, dbl_ms * 1000.0 / iterations);

    // scalar multiplication
    int scalar_iters = 100;
    Fr scalar = fr_from_u64(0xdeadbeef12345678ULL);
    timer.start();
    for (int i = 0; i < scalar_iters; ++i) {
        p = scalar_mul(G0, scalar);
    }
    double smul_ms = timer.elapsed_ms();
    printf("  scalar_mul:   %d ops in %.2f ms  (%.2f ms/op)\n",
           scalar_iters, smul_ms, smul_ms / scalar_iters);

    // prevent DCE
    if (point_is_identity(p)) printf("(prevent DCE)\n");
}

#ifdef __CUDACC__
void fill_benchmark_scalars(Fr* scalars, int batch_count) {
    for (int batch = 0; batch < batch_count; ++batch) {
        uint64_t state = 0x9e3779b97f4a7c15ULL ^ static_cast<uint64_t>(batch);
        for (int i = 0; i < MSM_SIZE; ++i) {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            scalars[batch * MSM_SIZE + i] =
                fr_from_u64(state * 0x2545f4914f6cdd1dULL);
        }
    }
}

void bench_msm_gpu_pippenger() {
    constexpr int batch_count = 64;
    constexpr int warmup_iterations = 3;
    constexpr int timed_iterations = 20;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        printf("\n[Benchmark] CUDA Pippenger MSM\n  SKIP: no CUDA device is available\n");
        return;
    }

    cudaDeviceProp device{};
    if (cudaGetDeviceProperties(&device, 0) != cudaSuccess) {
        printf("\n[Benchmark] CUDA Pippenger MSM\n  SKIP: unable to query CUDA device\n");
        return;
    }

    crs::CRSPoints crs_points;
    crs::load_crs(crs_points);
    MsmGpuContext context;
    MsmGpuWorkspace workspace;
    cuda_verkle::CudaPinnedBuffer<Fr> scalars;
    cuda_verkle::CudaPinnedBuffer<PointExtended> results;
    cuda_verkle::CudaStream stream;
    cuda_verkle::CudaEvent start;
    cuda_verkle::CudaEvent stop;

    MsmGpuStatus status = msm_gpu_context_init(context, crs_points.x, crs_points.y);
    if (status != MsmGpuStatus::success) goto cleanup;
    status = msm_gpu_workspace_init(workspace, batch_count);
    if (status != MsmGpuStatus::success) goto cleanup;
    if (scalars.allocate(static_cast<size_t>(batch_count) * MSM_SIZE) != cudaSuccess ||
        results.allocate(batch_count) != cudaSuccess ||
        stream.create(cudaStreamNonBlocking) != cudaSuccess ||
        start.create() != cudaSuccess || stop.create() != cudaSuccess) {
        status = MsmGpuStatus::cuda_error;
        goto cleanup;
    }
    fill_benchmark_scalars(scalars.data(), batch_count);

    for (int i = 0; i < warmup_iterations; ++i) {
        status = msm_gpu_compute_batch_async(
            context, workspace, scalars.data(), batch_count, results.data(), stream.get());
        if (status != MsmGpuStatus::success || msm_gpu_stream_synchronize(stream.get()) != MsmGpuStatus::success) {
            status = MsmGpuStatus::cuda_error;
            goto cleanup;
        }
    }

    // Events on the same stream measure host-to-device transfer, all Pippenger
    // kernels, and device-to-host transfer for each submitted batch.
    if (cudaEventRecord(start.get(), stream.get()) != cudaSuccess) {
        status = MsmGpuStatus::cuda_error;
        goto cleanup;
    }
    for (int i = 0; i < timed_iterations; ++i) {
        status = msm_gpu_compute_batch_async(
            context, workspace, scalars.data(), batch_count, results.data(), stream.get());
        if (status != MsmGpuStatus::success) goto cleanup;
    }
    if (cudaEventRecord(stop.get(), stream.get()) != cudaSuccess || cudaEventSynchronize(stop.get()) != cudaSuccess) {
        status = MsmGpuStatus::cuda_error;
        goto cleanup;
    }

    {
        float elapsed_ms = 0.0f;
        if (cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get()) != cudaSuccess) {
            status = MsmGpuStatus::cuda_error;
            goto cleanup;
        }
        PointExtended expected = msm_compute(scalars.data(), crs_points.x, crs_points.y, MSM_SIZE);
        bool matches_cpu = bw_eq({results.data()[0]}, {expected});
        const int total_msms = batch_count * timed_iterations;
        printf("\n[Benchmark] CUDA Pippenger MSM (256 points, w=8)\n");
        printf("  Device:                         %s (sm_%d%d)\n",
               device.name, device.major, device.minor);
        printf("  Batch size:                     %d MSMs\n", batch_count);
        printf("  Timed batches:                  %d\n", timed_iterations);
        printf("  End-to-end elapsed time:        %.2f ms\n", elapsed_ms);
        printf("  Average per MSM:                %.4f ms\n", elapsed_ms / total_msms);
        printf("  End-to-end throughput:          %.0f MSM/s\n",
               total_msms * 1000.0 / elapsed_ms);
        printf("  First result matches CPU:       %s\n", matches_cpu ? "yes" : "NO");
        if (!matches_cpu) status = MsmGpuStatus::cuda_error;
    }

cleanup:
    if (status != MsmGpuStatus::success) {
        printf("\n[Benchmark] CUDA Pippenger MSM\n  FAILED: GPU setup or execution error (%d)\n",
               static_cast<int>(status));
    }
    (void)msm_gpu_workspace_destroy(workspace);
    (void)msm_gpu_context_destroy(context);
}
#else
void bench_msm_gpu_pippenger() {
    printf("\n[Benchmark] CUDA Pippenger MSM\n");
    printf("  SKIP: rebuild this target with nvcc on an NVIDIA GPU to measure CUDA execution.\n");
}
#endif

int main() {
    printf("cuda-verkle MSM Benchmarks\n");

    bench_field_ops();
    bench_point_ops();
    bench_msm_256_pippenger();
    bench_msm_256_naive();
    bench_msm_sparse();
    bench_msm_gpu_pippenger();

    printf("\nSummary:\n");
    printf("  CPU baselines provide a reference for GPU Pippenger measurements.\n");

    return 0;
}
