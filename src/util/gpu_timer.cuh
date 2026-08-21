#pragma once
#include <cuda_runtime.h>
#include <cstdio>

class GpuTimer {
public:
    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }
    
    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
    }
    
    float elapsed_ms() {
        cudaEventSynchronize(stop_);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
    
private:
    cudaEvent_t start_, stop_;
};
