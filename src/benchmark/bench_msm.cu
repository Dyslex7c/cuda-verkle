// bench_msm.cu - MSM and commitment benchmarks
// Measures timing for the core MSM operation at various scales.
// On CPU: uses <chrono> for timing.
// CUDA execution is not implemented yet; these are CPU baselines.

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
#include "../src/msm/msm_kernel.cu"

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
    for (int i = 0; i < 256; ++i) scalars[i] = FR_ZERO;
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

int main() {
    printf("cuda-verkle CPU Benchmarks (CPU implementation)\n");

    bench_field_ops();
    bench_point_ops();
    bench_msm_256_pippenger();
    bench_msm_256_naive();
    bench_msm_sparse();

    printf("\nSummary:\n");
    printf("  These CPU baselines establish the performance floor.\n");
    printf("  No GPU acceleration is implemented in this benchmark.\n");

    return 0;
}
