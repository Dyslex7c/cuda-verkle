// Small, non-throwing RAII wrappers for CUDA runtime resources.
#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cstddef>
#include <limits>

namespace cuda_verkle {

class CudaStream final {
public:
    CudaStream() = default;
    ~CudaStream() { reset(); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&&) = delete;
    CudaStream& operator=(CudaStream&&) = delete;

    cudaError_t create(unsigned int flags = cudaStreamDefault) {
        const cudaError_t cleanup = reset();
        if (cleanup != cudaSuccess) return cleanup;
        return cudaStreamCreateWithFlags(&stream_, flags);
    }
    cudaError_t reset() {
        if (stream_ == nullptr) return cudaSuccess;
        const cudaError_t status = cudaStreamDestroy(stream_);
        stream_ = nullptr;
        return status;
    }
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

class CudaEvent final {
public:
    CudaEvent() = default;
    ~CudaEvent() { reset(); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    CudaEvent(CudaEvent&&) = delete;
    CudaEvent& operator=(CudaEvent&&) = delete;

    cudaError_t create(unsigned int flags = cudaEventDefault) {
        const cudaError_t cleanup = reset();
        if (cleanup != cudaSuccess) return cleanup;
        return cudaEventCreateWithFlags(&event_, flags);
    }
    cudaError_t reset() {
        if (event_ == nullptr) return cudaSuccess;
        const cudaError_t status = cudaEventDestroy(event_);
        event_ = nullptr;
        return status;
    }
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

template <typename T>
class CudaPinnedBuffer final {
public:
    CudaPinnedBuffer() = default;
    ~CudaPinnedBuffer() { reset(); }
    CudaPinnedBuffer(const CudaPinnedBuffer&) = delete;
    CudaPinnedBuffer& operator=(const CudaPinnedBuffer&) = delete;
    CudaPinnedBuffer(CudaPinnedBuffer&&) = delete;
    CudaPinnedBuffer& operator=(CudaPinnedBuffer&&) = delete;

    cudaError_t allocate(size_t count) {
        const cudaError_t cleanup = reset();
        if (cleanup != cudaSuccess) return cleanup;
        if (count == 0) return cudaSuccess;
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) return cudaErrorInvalidValue;
        const cudaError_t status = cudaMallocHost(reinterpret_cast<void**>(&data_), count * sizeof(T));
        if (status == cudaSuccess) count_ = count;
        return status;
    }
    cudaError_t reset() {
        if (data_ == nullptr) return cudaSuccess;
        const cudaError_t status = cudaFreeHost(data_);
        data_ = nullptr;
        count_ = 0;
        return status;
    }
    T* data() const { return data_; }
    size_t size() const { return count_; }

private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

} // namespace cuda_verkle
#endif
