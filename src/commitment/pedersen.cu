#include "pedersen.cuh"

BanderwagonElement PedersenCommitment::commit(const Fr values[], int n) const {
    // Do not read uninitialized CRS storage if a caller forgot init().
    if (!initialized || n <= 0) return BanderwagonElement{point_identity()};
    // Clamp n to CRS size
    if (n > crs::CRS_SIZE) n = crs::CRS_SIZE;
    
    // Use the MSM engine to compute sum(values[i] * G_i)
    PointExtended result = msm_compute(values, crs.x, crs.y, n);
    return BanderwagonElement{result};
}

void PedersenCommitment::commit_to_bytes(const Fr values[], int n, uint8_t out[32]) const {
    BanderwagonElement c = commit(values, n);
    bw_to_bytes(c, out);
}
