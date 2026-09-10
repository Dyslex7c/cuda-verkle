# Independent cryptographic audit release gate

No independent cryptographic audit has been performed for this repository.
Passing CI, static scans, and reference-vector tests does not change that.

Before any release is described as suitable for production use, commission an
independent auditor with experience in elliptic-curve and proof-system code.
The engagement must cover at least:

- Montgomery field arithmetic, reductions, canonical encodings, and inversion
  behavior;
- Bandersnatch/Banderwagon group arithmetic, subgroup validation, and
  serialization compatibility;
- CRS provenance and fixed-basis MSM correctness, including the CUDA kernels;
- IPA transcript domain separation, challenge derivation, proof parsing, and
  verification equations;
- Verkle state-tree construction, persistence parsing, and EIP-6800 mapping;
- malformed-input handling, memory/resource lifetime, integer overflow, and
  denial-of-service limits; and
- deterministic CPU/GPU equivalence on supported hardware and CUDA versions.

The audit report, resolved findings, exact audited commit, scope exclusions,
and any residual risks must be published before changing the project-status
statement in `SECURITY.md` or the README.
