# CUDA Ethereum Verkle trees

A C++/CUDA research implementation of Pedersen vector commitments over the Banderwagon group, targeting Ethereum's Verkle tree proposal ([EIP-6800](https://eips.ethereum.org/EIPS/eip-6800)).

Covers the core commitment stack: Montgomery field arithmetic → twisted Edwards curve operations → Pippenger multi-scalar multiplication → Pedersen commitments → 256-element IPA opening proofs → an EIP-6800 sparse key/value state tree. The tree implements extension/suffix nodes, absent-vs-zero leaf encoding, recursive main-tree commitments, EIP-6800 `group_to_scalar_field`, and Pedersen state-key derivation. It remains experimental cryptographic software, not production-ready.

The state tree exposes `set`, `get`, `erase`, `root`, `serialize`/`deserialize`, and `save`/`load`. Its versioned persistence format stores canonical, lexicographically ordered key/value records and reconstructs the branch/extension topology on load; malformed or non-canonical snapshots are rejected without modifying the loaded tree.

Serialized public inputs must use the strict decoding APIs: `fr_from_bytes_strict` accepts only canonical 32-byte big-endian scalars, while `bw_from_bytes_strict` additionally recovers the curve point and rejects off-curve and non-subgroup Banderwagon encodings. These validation routines are variable-time and must not be used with secret inputs.

> **Status:** Host unit, property, differential, IPA, and EIP-6800 state-tree tests pass. A windowed CUDA Pippenger MSM with batched, stream-aware execution, GPU integration tests, and CUDA-event benchmarking is included. CI builds host and CUDA targets and scans committed secrets and Rust dependencies; NVIDIA hardware validation and an independent cryptographic audit remain required before any production claim.

---

## Why and what's this about

Ethereum's state transition requires recomputing Pedersen commitments over 256-wide vectors on every block. Each commitment is a multi-scalar multiplication (MSM) of 256 scalars against a fixed basis on the Banderwagon curve. Every existing implementation ([rust-verkle](https://github.com/crate-crypto/rust-verkle), [go-verkle](https://github.com/crate-crypto/go-ipa), [constantine](https://github.com/mratsim/constantine)) runs on CPU. This project explores GPU acceleration of that inner loop.

---

## How to Run

### Option 1: On CPU / Mac / Linux (No GPU or CUDA Required)

Any C++17 compiler (`clang++` or `g++`) works directly:

```bash
git clone <this-repo> && cd cuda-verkle

# Run host unit, property, differential, and state-tree tests
c++ -std=c++17 -x c++ -O2 -I src -o test_field tests/test_field.cu && ./test_field
c++ -std=c++17 -x c++ -O2 -I src -o test_curve tests/test_curve.cu && ./test_curve
c++ -std=c++17 -x c++ -O2 -I src -o test_msm tests/test_msm.cu src/msm/msm_kernel.cu && ./test_msm
c++ -std=c++17 -x c++ -O2 -I src -o test_commitment tests/test_commitment.cu src/msm/msm_kernel.cu src/commitment/pedersen.cu && ./test_commitment
c++ -std=c++17 -x c++ -O2 -I src -o test_tree tests/test_tree.cu src/msm/msm_kernel.cu && ./test_tree
c++ -std=c++17 -x c++ -O2 -I src -o test_properties tests/test_properties.cu src/msm/msm_kernel.cu && ./test_properties
c++ -std=c++17 -x c++ -O2 -I src -o test_ipa tests/test_ipa.cu src/msm/msm_kernel.cu && ./test_ipa

# Run CPU benchmarks
c++ -std=c++17 -x c++ -O2 -I src -o bench src/benchmark/bench_msm.cu src/msm/msm_kernel.cu && ./bench
```

---

### Option 2: On an NVIDIA GPU with `nvcc`

If you have an NVIDIA GPU and the CUDA Toolkit installed:

#### Direct `nvcc` Compilation:
```bash
# Compile and run test suite with nvcc
nvcc -std=c++17 -O3 -I src -arch=sm_75 -o test_field tests/test_field.cu && ./test_field
nvcc -std=c++17 -O3 -I src -arch=sm_75 -o test_curve tests/test_curve.cu && ./test_curve
nvcc -std=c++17 -O3 -I src -arch=sm_75 -o test_msm tests/test_msm.cu src/msm/msm_kernel.cu && ./test_msm
nvcc -std=c++17 -O3 -I src -arch=sm_75 -o test_commitment tests/test_commitment.cu src/msm/msm_kernel.cu src/commitment/pedersen.cu && ./test_commitment
nvcc -std=c++17 -O3 -I src -arch=sm_75 -o test_tree tests/test_tree.cu src/msm/msm_kernel.cu && ./test_tree
nvcc -std=c++17 -O3 -I src -arch=sm_75 -o test_msm_gpu tests/test_msm_gpu.cu src/msm/msm_kernel.cu && ./test_msm_gpu
nvcc -std=c++17 -O3 -I src -arch=sm_75 -o test_ipa tests/test_ipa.cu src/msm/msm_kernel.cu && ./test_ipa

# Compile and run benchmarks
nvcc -std=c++17 -O3 -I src -arch=sm_75 -o bench src/benchmark/bench_msm.cu src/msm/msm_kernel.cu && ./bench
```

> **Target Architecture (`-arch=sm_XX`):**
> - `sm_75` — Tesla T4 / Turing (Google Colab free tier)
> - `sm_80` — A100 / Ampere
> - `sm_86` — RTX 3080 / 3090
> - `sm_89` — RTX 4080 / 4090 / Ada Lovelace

#### CMake Build:
```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75
make -j$(nproc)
ctest --output-on-failure
```

#### Running on Google Colab (Free GPU):
1. Open a new notebook on [Google Colab](https://colab.research.google.com).
2. Set runtime to **GPU** (`Runtime` → `Change runtime type` → `T4 GPU`).
3. Clone and build:
   ```bash
   !git clone https://github.com/<your-username>/cuda-verkle.git
   %cd cuda-verkle
   !mkdir -p build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 && make -j$(nproc)
   !cd build && ctest --output-on-failure
   ```

#### With Docker (GPU Passthrough):
```bash
docker compose build
docker compose run --rm cuda-verkle bash -c "cd build && ctest --output-on-failure"
```

---

## Tests

The host suites include deterministic known-answer tests generated by the locked Rust reference dependencies: field arithmetic, 256-wide Pedersen commitments, logarithmic 256-element IPA opening proofs, and depth-1 incremental tree updates. IPA tests compare the exact compressed proof encoding and verify a proof generated independently by the Rust reference. They complement algebraic identities, on-curve verification of CRS points, CPU/GPU-windowed Pippenger cross-validation, commitment homomorphism, and incremental-vs-full tree recomputation.

`src/proof/ipa.cuh` implements the transparent Fiat–Shamir inner-product argument used by the pinned `ipa-multipoint` reference: one commitment evaluation is proven with 8 L/R rounds (544 serialized bytes). `ipa_proof_from_bytes_strict` accepts only that exact format and rejects truncated input, non-canonical scalars, malformed curve encodings, and points outside the Banderwagon subgroup before verification.

The fixtures in [`test_vectors/`](test_vectors/) are versioned, deterministic outputs from `rust-reference/`, which uses the pinned [rust-verkle](https://github.com/crate-crypto/rust-verkle) dependencies. Regenerate them with `make vectors` (or `cd rust-reference && cargo run --locked --release -- generate`), then review and commit the JSON diff. C++ tests fail if their required fixture is missing or malformed.

## Benchmarking

`bench_msm` reports CPU reference timings on every platform. When built with
`nvcc` and run on an NVIDIA GPU, it additionally measures batched GPU
Pippenger MSMs with CUDA events. The GPU figure is end-to-end: pinned-host
input transfer, scalar conversion, all window kernels, result transfer, and
stream completion. It reports batch size, average time per 256-point MSM,
throughput, device name, and a CPU cross-check of the first result.

## Curve parameters

| Param | Value |
|-|-------|
| Base field Fp | BLS12-381 scalar field: p = `0x73eda753299d7d48…00000001` |
| Scalar field Fr | Bandersnatch subgroup order: n = `0x1cfb69d4ca675f52…2876e7e1` |
| Curve | Twisted Edwards: −5x² + y² = 1 + dx²y² |
| Coordinates | Extended projective (X : Y : T : Z), T = XY/Z |
| CRS | 256 generators + Q, seed `eth_verkle_oct_2021` |

## Design notes

**Montgomery arithmetic.** Every field element is stored as a·R mod p. Multiplication uses the CIOS (Coarsely Integrated Operand Scanning) algorithm with 8 rounds of multiply-accumulate-reduce. The GPU path emits PTX `mad.lo.cc.u32` / `madc.hi.cc.u32` carry chains; the host path uses `uint64_t` widening. The n′ constant for Fp happens to be 0xFFFFFFFF, which makes the reduction step a simple multiply-by-minus-one.

**Extended projective coordinates.** Storing (X, Y, T, Z) with T = XY/Z trades one extra field element per point for elimination of all inversions during addition (8M + 1D) and doubling (4S + 3M). Inversions only happen during final affine conversion for serialization.

**Struct-of-Arrays CRS.** The 256 basis points are stored as `x[256]`, `y[256]` rather than `{x,y}[256]`. On GPU, this means consecutive threads in a warp read consecutive memory addresses — coalesced access at full bandwidth.

**Incremental recommitment.** When leaf i changes from v to v′, the tree updates the parent commitment via C′ = C + (v′ − v) · Gᵢ — a single scalar multiplication instead of a full 256-wide MSM. This is the operation that dominates Ethereum block processing at scale.

## References

- [EIP-6800](https://eips.ethereum.org/EIPS/eip-6800) — Ethereum state using Verkle trees
- [rust-verkle](https://github.com/crate-crypto/rust-verkle) — Canonical Rust implementation
- [Bandersnatch](https://eprint.iacr.org/2021/1152) — Masson, Sanso, Zhang (2021)
- [Pippenger](https://cr.yp.to/papers/pippenger.pdf) — Bucket method for multi-scalar multiplication

## License

This project is licensed under the [MIT License](./LICENSE).
See [third-party notices](./THIRD_PARTY_NOTICES.md) for CRS-data provenance.
