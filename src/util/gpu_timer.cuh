#pragma once
#include "cuda_raii.cuh"

class GpuTimer final {
public:
    GpuTimer() : status_(start_.create()) {
        if (status_ == cudaSuccess) status_ = stop_.create();
    }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    bool valid() const { return status_ == cudaSuccess; }
    cudaError_t status() const { return status_; }

    cudaError_t start(cudaStream_t stream = 0) {
        if (!valid()) return status_;
        return cudaEventRecord(start_.get(), stream);
    }

    cudaError_t stop(cudaStream_t stream = 0) {
        if (!valid()) return status_;
        return cudaEventRecord(stop_.get(), stream);
    }

    cudaError_t elapsed_ms(float& ms) {
        if (!valid()) return status_;
        cudaError_t result = cudaEventSynchronize(stop_.get());
        if (result != cudaSuccess) return result;
        return cudaEventElapsedTime(&ms, start_.get(), stop_.get());
    }

    // Compatibility convenience for callers that do not need the CUDA error.
    float elapsed_ms() {
        float ms = 0.0f;
        (void)elapsed_ms(ms);
        return ms;
    }

private:
    cuda_verkle::CudaEvent start_;
    cuda_verkle::CudaEvent stop_;
    cudaError_t status_ = cudaSuccess;
};
