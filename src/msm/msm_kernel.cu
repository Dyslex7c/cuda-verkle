#include "msm_kernel.cuh"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

// Extract bits [window_idx*w, window_idx*w + w) from the scalar
__host__ __device__ inline uint32_t get_window(const Fr& scalar, int window_idx, int w) {
    uint32_t raw[8];
    fr_to_raw(scalar, raw);
    int bit_start = window_idx * w;
    int limb_idx = bit_start / 32;
    int bit_offset = bit_start % 32;
    uint32_t mask = (1u << w) - 1;
    uint32_t digit = (raw[limb_idx] >> bit_offset) & mask;
    // Handle spanning two limbs
    if (bit_offset + w > 32 && limb_idx + 1 < 8) {
        digit |= (raw[limb_idx + 1] << (32 - bit_offset)) & mask;
    }
    return digit;
}

// for each (scalar, point), compute scalar_mul and accumulate
__host__ __device__ PointExtended msm_naive(const Fr scalars[], const Fp px[], const Fp py[], int n) {
    PointExtended result = point_identity();
    for (int i = 0; i < n; ++i) {
        if (fr_is_zero(scalars[i])) continue;  // skip zero scalars
        PointAffine base = {px[i], py[i]};
        PointExtended base_ext = point_from_affine(base);
        PointExtended term = scalar_mul(base_ext, scalars[i]);
        result = point_add(result, term);
    }
    return result;
}

__host__ __device__ PointExtended msm_cpu_reference(
    const Fr scalars[],
    const Fp point_x[],
    const Fp point_y[],
    int n) {
    return msm_naive(scalars, point_x, point_y, n);
}

__host__ __device__ PointExtended msm_compute(
    const Fr scalars[],
    const Fp point_x[],
    const Fp point_y[],
    int n) {
    // The CRS has a fixed width. This is a public function, so protect callers from indexing past the supplied fixed-size point arrays.
    if (n <= 0) return point_identity();
    if (n > MSM_SIZE) n = MSM_SIZE;
    
    int w = 8;
    int num_windows = (253 + w - 1) / w; // ceil(253 / 8) = 32
    
    PointExtended total = point_identity();
    
    // For w=8, 2^w = 256
    PointExtended buckets[256];

    for (int window_idx = num_windows - 1; window_idx >= 0; --window_idx) {
        // Shift total by w bits
        for (int i = 0; i < w; ++i) {
            total = point_double(total);
        }
        
        // Initialize buckets to identity
        for (int i = 1; i < 256; ++i) {
            buckets[i] = point_identity();
        }
        
        for (int i = 0; i < n; ++i) {
            uint32_t digit = get_window(scalars[i], window_idx, w);
            if (digit != 0) {
                PointAffine base = {point_x[i], point_y[i]};
                PointExtended base_ext = point_from_affine(base);
                buckets[digit] = point_add(buckets[digit], base_ext);
            }
        }
        
        PointExtended running_sum = point_identity();
        PointExtended partial = point_identity();
        
        for (int j = 255; j >= 1; --j) {
            running_sum = point_add(running_sum, buckets[j]);
            partial = point_add(partial, running_sum);
        }
        
        total = point_add(total, partial);
    }
    
    return total;
}

#ifdef __CUDACC__
namespace {

// One thread computes one scalar multiplication.  The 256 partial points are
// then reduced in shared memory.  This avoids cross-thread mutation of a point
// and is deterministic for a fixed input, unlike an atomic bucket accumulator.
// It is deliberately a correctness baseline; a future optimized kernel can
// replace the per-thread scalar multiplication with parallel Pippenger windows
// without changing the host API.
__global__ void msm_gpu_baseline_kernel(
    const Fr* scalars,
    const Fp* point_x,
    const Fp* point_y,
    int n,
    PointExtended* result) {
    __shared__ PointExtended partials[MSM_SIZE];

    const int tid = static_cast<int>(threadIdx.x);
    PointExtended partial = point_identity();
    if (tid < n && !fr_is_zero(scalars[tid])) {
        PointAffine base = {point_x[tid], point_y[tid]};
        partial = scalar_mul(point_from_affine(base), scalars[tid]);
    }
    partials[tid] = partial;
    __syncthreads();

    for (int stride = MSM_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] = point_add(partials[tid], partials[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) *result = partials[0];
}

MsmGpuStatus status_from_cuda(cudaError_t error) {
    if (error == cudaSuccess) return MsmGpuStatus::success;
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver) {
        return MsmGpuStatus::no_cuda_device;
    }
    return MsmGpuStatus::cuda_error;
}

void clear_context(MsmGpuContext& context) {
    context.device_scalars = nullptr;
    context.device_point_x = nullptr;
    context.device_point_y = nullptr;
    context.device_result = nullptr;
    context.initialized = false;
}

} // namespace

MsmGpuStatus msm_gpu_context_init(
    MsmGpuContext& context,
    const Fp point_x[MSM_SIZE],
    const Fp point_y[MSM_SIZE]) {
    if (point_x == nullptr || point_y == nullptr) {
        return MsmGpuStatus::invalid_argument;
    }

    MsmGpuStatus destroy_status = msm_gpu_context_destroy(context);
    if (destroy_status != MsmGpuStatus::success) return destroy_status;

    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        return error == cudaSuccess ? MsmGpuStatus::no_cuda_device : status_from_cuda(error);
    }

    error = cudaMalloc(&context.device_scalars, MSM_SIZE * sizeof(Fr));
    if (error != cudaSuccess) goto fail;
    error = cudaMalloc(&context.device_point_x, MSM_SIZE * sizeof(Fp));
    if (error != cudaSuccess) goto fail;
    error = cudaMalloc(&context.device_point_y, MSM_SIZE * sizeof(Fp));
    if (error != cudaSuccess) goto fail;
    error = cudaMalloc(&context.device_result, sizeof(PointExtended));
    if (error != cudaSuccess) goto fail;

    error = cudaMemcpy(context.device_point_x, point_x, MSM_SIZE * sizeof(Fp), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) goto fail;
    error = cudaMemcpy(context.device_point_y, point_y, MSM_SIZE * sizeof(Fp), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) goto fail;

    context.initialized = true;
    return MsmGpuStatus::success;

fail:
    const MsmGpuStatus status = status_from_cuda(error);
    (void)msm_gpu_context_destroy(context);
    return status;
}

MsmGpuStatus msm_gpu_compute(
    MsmGpuContext& context,
    const Fr scalars[],
    int n,
    PointExtended* result) {
    if (!context.initialized) return MsmGpuStatus::not_initialized;
    if (result == nullptr || n < 0 || n > MSM_SIZE || (n > 0 && scalars == nullptr)) {
        return MsmGpuStatus::invalid_argument;
    }
    if (n == 0) {
        *result = point_identity();
        return MsmGpuStatus::success;
    }

    cudaError_t error = cudaMemcpy(
        context.device_scalars, scalars, static_cast<size_t>(n) * sizeof(Fr), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return status_from_cuda(error);

    msm_gpu_baseline_kernel<<<1, MSM_SIZE>>>(
        static_cast<const Fr*>(context.device_scalars),
        static_cast<const Fp*>(context.device_point_x),
        static_cast<const Fp*>(context.device_point_y),
        n,
        static_cast<PointExtended*>(context.device_result));
    error = cudaGetLastError();
    if (error != cudaSuccess) return status_from_cuda(error);
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) return status_from_cuda(error);
    error = cudaMemcpy(result, context.device_result, sizeof(PointExtended), cudaMemcpyDeviceToHost);
    return status_from_cuda(error);
}

MsmGpuStatus msm_gpu_context_destroy(MsmGpuContext& context) {
    cudaError_t first_error = cudaSuccess;
    void* allocations[] = {
        context.device_scalars,
        context.device_point_x,
        context.device_point_y,
        context.device_result,
    };
    for (void* allocation : allocations) {
        if (allocation == nullptr) continue;
        cudaError_t error = cudaFree(allocation);
        if (first_error == cudaSuccess && error != cudaSuccess) first_error = error;
    }
    clear_context(context);
    return status_from_cuda(first_error);
}
#endif
