#!/usr/bin/env bash
# Reproducible CUDA validation for a machine with an NVIDIA GPU.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build-gpu}"
cuda_arch="${CUDA_ARCH:-75}"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc is required. Install the CUDA Toolkit or use a GPU runtime." >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is required. Select an NVIDIA GPU runtime first." >&2
    exit 1
fi

echo "GPU environment:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvcc --version | tail -n 1
if git -C "${repo_root}" rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "Commit: $(git -C "${repo_root}" rev-parse HEAD)"
fi
echo "Configuring for CUDA architecture sm_${cuda_arch}"

cmake -S "${repo_root}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}" \
    -DENABLE_PHASE2=ON \
    -DENABLE_PHASE3=ON \
    -DENABLE_PHASE4=ON
cmake --build "${build_dir}" --parallel
ctest --test-dir "${build_dir}" --output-on-failure
"${build_dir}/bench_msm"

echo "GPU validation completed successfully. Save the test and benchmark output with the commit SHA."
