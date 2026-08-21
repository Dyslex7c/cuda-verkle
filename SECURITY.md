# Security Policy

## Project status

This is experimental cryptographic research software. It has not received an
independent security audit and must not be used to secure funds, consensus,
production state, or other high-value data.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Contact the
maintainer privately through the GitHub account associated with this repository
and include a minimal reproduction, impact assessment, and affected revision.

## Required pre-publication audit

Do not state that this repository is safe to publish, push, or make public
until the following checks have been performed and their results reported.
This applies to human contributors and automated/LLM-assisted workflows.

1. Scan all files that will be committed for credentials and private material:
   API keys, tokens, passwords, private keys, seed phrases, `.env` files, and
   cloud credential files.
2. Inspect the publish set for generated artifacts, binaries, build caches,
   crash dumps, logs, editor metadata, and files containing absolute local
   paths. Remove them or ensure they are ignored.
3. Verify `.gitignore` covers the project’s CMake/CUDA, Rust, editor, and local
   environment outputs.
4. Review third-party code, datasets, CRS data, and dependencies for license,
   attribution, and redistribution obligations. Do not add a project license
   unless the contributor is authorized to grant it for all included material.
5. Confirm documentation accurately describes the implementation’s security
   and verification status. Do not claim a GPU implementation, protocol
   compatibility, external audit, or production readiness without evidence.
6. Run the available tests and report any tests that could not run, including
   the reason (for example, unavailable CUDA hardware/toolkit or network).
7. Review the final staged file list with `git status --short` and
   `git diff --cached --name-only` before pushing.

Passing this checklist reduces accidental disclosure risk; it is not a
cryptographic security audit or a guarantee that the project is safe to use.
