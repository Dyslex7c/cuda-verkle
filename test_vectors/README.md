# Test Vectors for CUDA Verkle

This directory contains test vectors for verifying the implementations of finite field operations, curve operations, multi-scalar multiplication (MSM), and the Verkle tree commitment computations.

## Format

The test vectors are provided in a simplified line-by-line or minimal JSON-like format. Each hex string represents a 256-bit integer (typically little-endian limbs inside the engine, but provided here in standard big-endian hex notation, e.g., `0x1a2b...`).

## Regenerating Test Vectors

To regenerate the test vectors, use the Rust reference implementation included in `rust-reference/`.

```bash
cd ../rust-reference
cargo run --release -- generate
```

This will run the Rust executable, perform the reference calculations using established Rust cryptographic libraries, and overwrite the test vector files in this directory with the new known-answer tests.
