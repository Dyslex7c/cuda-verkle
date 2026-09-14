// fp.cu is intentionally minimal as all Fp field arithmetic is implemented via inline __host__ __device__ functions in fp.cuh to ensure performance and avoid CUDA linkage overhead.

#include "fp.cuh"
