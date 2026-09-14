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
    
    const int w = MSM_WINDOW_BITS;
    const int num_windows = MSM_NUM_WINDOWS;
    
    PointExtended total = point_identity();
    
    PointExtended buckets[MSM_BUCKET_COUNT];

    for (int window_idx = num_windows - 1; window_idx >= 0; --window_idx) {
        // Shift total by w bits
        for (int i = 0; i < w; ++i) {
            total = point_double(total);
        }
        
        // Initialize buckets to identity
        for (int i = 1; i < MSM_BUCKET_COUNT; ++i) {
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
        
        for (int j = MSM_BUCKET_COUNT - 1; j >= 1; --j) {
            running_sum = point_add(running_sum, buckets[j]);
            partial = point_add(partial, running_sum);
        }
        
        total = point_add(total, partial);
    }
    
    return total;
}

#ifdef __CUDACC__
namespace {

// Convert each scalar out of Montgomery form once. The Pippenger window
// kernels then read raw scalar limbs directly instead of performing an inverse
// Montgomery reduction for every scalar/window/bucket comparison.
__global__ void scalar_to_raw_kernel(
    const Fr* scalars,
    uint32_t* raw_scalars) {
    const int tid = static_cast<int>(threadIdx.x);
    const int batch_index = static_cast<int>(blockIdx.x);
    const int scalar_index = batch_index * MSM_SIZE + tid;
    fr_to_raw(scalars[scalar_index], raw_scalars + scalar_index * 8);
}

__device__ inline uint32_t get_raw_window(
    const uint32_t* raw_scalars,
    int scalar_index,
    int window_index) {
    // MSM_WINDOW_BITS divides 32, so an 8-bit window never spans limbs.
    const int bit_start = window_index * MSM_WINDOW_BITS;
    const int limb_index = bit_start / 32;
    const int bit_offset = bit_start % 32;
    return (raw_scalars[scalar_index * 8 + limb_index] >> bit_offset) &
           (MSM_BUCKET_COUNT - 1);
}

// Grid: one block per scalar window; block: one thread per bucket.  Each
// bucket is owned by exactly one thread, eliminating point-addition races.
// The 32 window blocks run concurrently and write their weighted bucket sums
// for the final Horner-style window combination kernel.
__global__ void msm_gpu_pippenger_windows_kernel(
    const uint32_t* raw_scalars,
    const Fp* point_x,
    const Fp* point_y,
    PointExtended* window_sums) {
    __shared__ PointExtended buckets[MSM_BUCKET_COUNT];

    const int tid = static_cast<int>(threadIdx.x);
    const int window_index = static_cast<int>(blockIdx.x);
    const int batch_index = static_cast<int>(blockIdx.y);
    const int scalar_offset = batch_index * MSM_SIZE;

    PointExtended bucket = point_identity();
    if (tid != 0) {
        for (int scalar_index = 0; scalar_index < MSM_SIZE; ++scalar_index) {
            const int global_scalar_index = scalar_offset + scalar_index;
            if (get_raw_window(raw_scalars, global_scalar_index, window_index) ==
                static_cast<uint32_t>(tid)) {
                PointAffine base = {point_x[scalar_index], point_y[scalar_index]};
                bucket = point_add(bucket, point_from_affine(base));
            }
        }
    }
    buckets[tid] = bucket;
    __syncthreads();

    if (tid == 0) {
        PointExtended running_sum = point_identity();
        PointExtended weighted_sum = point_identity();
        for (int bucket_index = MSM_BUCKET_COUNT - 1; bucket_index >= 1; --bucket_index) {
            running_sum = point_add(running_sum, buckets[bucket_index]);
            weighted_sum = point_add(weighted_sum, running_sum);
        }
        window_sums[batch_index * MSM_NUM_WINDOWS + window_index] = weighted_sum;
    }
}

// Combine sum_j (window_sum[j] * 2^(j * MSM_WINDOW_BITS)) from the most
// significant window down. The small serial tail is intentional: each window
// sum has already been constructed by an independent CUDA block.
__global__ void msm_gpu_combine_windows_kernel(
    const PointExtended* window_sums,
    PointExtended* result) {
    if (threadIdx.x != 0) return;
    const int batch_index = static_cast<int>(blockIdx.x);
    const PointExtended* batch_window_sums =
        window_sums + batch_index * MSM_NUM_WINDOWS;

    PointExtended total = batch_window_sums[MSM_NUM_WINDOWS - 1];
    for (int window_index = MSM_NUM_WINDOWS - 2; window_index >= 0; --window_index) {
        for (int bit = 0; bit < MSM_WINDOW_BITS; ++bit) {
            total = point_double(total);
        }
        total = point_add(total, batch_window_sums[window_index]);
    }
    result[batch_index] = total;
}

MsmGpuStatus status_from_cuda(cudaError_t error) {
    if (error == cudaSuccess) return MsmGpuStatus::success;
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver) {
        return MsmGpuStatus::no_cuda_device;
    }
    return MsmGpuStatus::cuda_error;
}

void clear_workspace(MsmGpuWorkspace& workspace) {
    workspace.device_scalars = nullptr;
    workspace.device_scalar_raw = nullptr;
    workspace.device_window_sums = nullptr;
    workspace.device_result = nullptr;
    workspace.batch_capacity = 0;
    workspace.initialized = false;
}

void clear_context(MsmGpuContext& context) {
    context.device_point_x = nullptr;
    context.device_point_y = nullptr;
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

    error = cudaMalloc(&context.device_point_x, MSM_SIZE * sizeof(Fp));
    if (error != cudaSuccess) goto fail;
    error = cudaMalloc(&context.device_point_y, MSM_SIZE * sizeof(Fp));
    if (error != cudaSuccess) goto fail;

    error = cudaMemcpy(context.device_point_x, point_x, MSM_SIZE * sizeof(Fp), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) goto fail;
    error = cudaMemcpy(context.device_point_y, point_y, MSM_SIZE * sizeof(Fp), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) goto fail;

    {
        const MsmGpuStatus workspace_status =
            msm_gpu_workspace_init(context.default_workspace, 1);
        if (workspace_status != MsmGpuStatus::success) {
            (void)msm_gpu_context_destroy(context);
            return workspace_status;
        }
    }
    context.initialized = true;
    return MsmGpuStatus::success;

fail:
    const MsmGpuStatus status = status_from_cuda(error);
    (void)msm_gpu_context_destroy(context);
    return status;
}

MsmGpuStatus msm_gpu_workspace_init(MsmGpuWorkspace& workspace, int batch_capacity) {
    if (batch_capacity <= 0 || batch_capacity > 65535) {
        return MsmGpuStatus::invalid_argument;
    }
    MsmGpuStatus destroy_status = msm_gpu_workspace_destroy(workspace);
    if (destroy_status != MsmGpuStatus::success) return destroy_status;

    cudaError_t error = cudaMalloc(
        &workspace.device_scalars, static_cast<size_t>(batch_capacity) * MSM_SIZE * sizeof(Fr));
    if (error != cudaSuccess) goto fail;
    error = cudaMalloc(
        &workspace.device_scalar_raw,
        static_cast<size_t>(batch_capacity) * MSM_SIZE * 8 * sizeof(uint32_t));
    if (error != cudaSuccess) goto fail;
    error = cudaMalloc(
        &workspace.device_window_sums,
        static_cast<size_t>(batch_capacity) * MSM_NUM_WINDOWS * sizeof(PointExtended));
    if (error != cudaSuccess) goto fail;
    error = cudaMalloc(
        &workspace.device_result, static_cast<size_t>(batch_capacity) * sizeof(PointExtended));
    if (error != cudaSuccess) goto fail;

    workspace.batch_capacity = batch_capacity;
    workspace.initialized = true;
    return MsmGpuStatus::success;

fail:
    const MsmGpuStatus status = status_from_cuda(error);
    (void)msm_gpu_workspace_destroy(workspace);
    return status;
}

MsmGpuStatus msm_gpu_workspace_destroy(MsmGpuWorkspace& workspace) {
    cudaError_t first_error = cudaSuccess;
    void* allocations[] = {
        workspace.device_scalars,
        workspace.device_scalar_raw,
        workspace.device_window_sums,
        workspace.device_result,
    };
    for (void* allocation : allocations) {
        if (allocation == nullptr) continue;
        cudaError_t error = cudaFree(allocation);
        if (first_error == cudaSuccess && error != cudaSuccess) first_error = error;
    }
    clear_workspace(workspace);
    return status_from_cuda(first_error);
}

MsmGpuStatus msm_gpu_compute_batch_async(
    MsmGpuContext& context,
    MsmGpuWorkspace& workspace,
    const Fr scalars[],
    int batch_count,
    PointExtended results[],
    cudaStream_t stream) {
    if (!context.initialized || !workspace.initialized) {
        return MsmGpuStatus::not_initialized;
    }
    if (batch_count <= 0 || batch_count > workspace.batch_capacity ||
        batch_count > 65535 || scalars == nullptr || results == nullptr) {
        return MsmGpuStatus::invalid_argument;
    }

    const size_t scalar_bytes = static_cast<size_t>(batch_count) * MSM_SIZE * sizeof(Fr);
    const size_t result_bytes = static_cast<size_t>(batch_count) * sizeof(PointExtended);
    cudaError_t error = cudaMemcpyAsync(
        workspace.device_scalars, scalars, scalar_bytes, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) return status_from_cuda(error);

    scalar_to_raw_kernel<<<batch_count, MSM_SIZE, 0, stream>>>(
        static_cast<const Fr*>(workspace.device_scalars),
        static_cast<uint32_t*>(workspace.device_scalar_raw));
    error = cudaGetLastError();
    if (error != cudaSuccess) return status_from_cuda(error);

    const dim3 window_grid(MSM_NUM_WINDOWS, static_cast<unsigned int>(batch_count));
    msm_gpu_pippenger_windows_kernel<<<window_grid, MSM_BUCKET_COUNT, 0, stream>>>(
        static_cast<const uint32_t*>(workspace.device_scalar_raw),
        static_cast<const Fp*>(context.device_point_x),
        static_cast<const Fp*>(context.device_point_y),
        static_cast<PointExtended*>(workspace.device_window_sums));
    error = cudaGetLastError();
    if (error != cudaSuccess) return status_from_cuda(error);

    msm_gpu_combine_windows_kernel<<<batch_count, 1, 0, stream>>>(
        static_cast<const PointExtended*>(workspace.device_window_sums),
        static_cast<PointExtended*>(workspace.device_result));
    error = cudaGetLastError();
    if (error != cudaSuccess) return status_from_cuda(error);

    error = cudaMemcpyAsync(
        results, workspace.device_result, result_bytes, cudaMemcpyDeviceToHost, stream);
    return status_from_cuda(error);
}

MsmGpuStatus msm_gpu_stream_synchronize(cudaStream_t stream) {
    return status_from_cuda(cudaStreamSynchronize(stream));
}

MsmGpuStatus msm_gpu_compute_batch(
    MsmGpuContext& context,
    MsmGpuWorkspace& workspace,
    const Fr scalars[],
    int batch_count,
    PointExtended results[]) {
    MsmGpuStatus status = msm_gpu_compute_batch_async(
        context, workspace, scalars, batch_count, results, 0);
    if (status != MsmGpuStatus::success) return status;
    return msm_gpu_stream_synchronize(0);
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

    // Batch operations are fixed-width. Pad a partial MSM with zero scalars
    // before dispatching it through the same Pippenger pipeline.
    if (n == MSM_SIZE) {
        return msm_gpu_compute_batch(
            context, context.default_workspace, scalars, 1, result);
    }
    Fr padded_scalars[MSM_SIZE];
    for (int i = 0; i < n; ++i) padded_scalars[i] = scalars[i];
    for (int i = n; i < MSM_SIZE; ++i) padded_scalars[i] = fr_zero();
    return msm_gpu_compute_batch(
        context, context.default_workspace, padded_scalars, 1, result);
}

MsmGpuStatus msm_gpu_context_destroy(MsmGpuContext& context) {
    cudaError_t first_error = cudaSuccess;
    MsmGpuStatus workspace_status = msm_gpu_workspace_destroy(context.default_workspace);
    if (workspace_status != MsmGpuStatus::success) {
        first_error = cudaErrorUnknown;
    }
    void* allocations[] = {
        context.device_point_x,
        context.device_point_y,
    };
    for (void* allocation : allocations) {
        if (allocation == nullptr) continue;
        cudaError_t error = cudaFree(allocation);
        if (first_error == cudaSuccess && error != cudaSuccess) first_error = error;
    }
    clear_context(context);
    return status_from_cuda(first_error);
}

MsmGpuWorkspace::~MsmGpuWorkspace() {
    (void)msm_gpu_workspace_destroy(*this);
}

MsmGpuContext::~MsmGpuContext() {
    (void)msm_gpu_context_destroy(*this);
}
#endif
