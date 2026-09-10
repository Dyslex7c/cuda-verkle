// Transparent Fiat-Shamir Inner Product Argument (IPA) for 256-element Pedersen commitments.
#pragma once

#include "../constants/crs_points.cuh"
#include "../curve/banderwagon.cuh"
#include "../field/fr.cuh"
#include "../msm/msm_kernel.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

static constexpr int IPA_WIDTH = 256;
static constexpr int IPA_ROUNDS = 8; // log2(256)
static constexpr size_t IPA_PROOF_BYTES = (2 * IPA_ROUNDS + 1) * 32;

namespace ipa_detail {

inline uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

// Minimal self-contained SHA-256 implementation for the Fiat-Shamir transcript
inline void sha256(const std::vector<uint8_t>& input, uint8_t out[32]) {
    static constexpr uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    std::vector<uint8_t> bytes(input);
    const uint64_t bits = static_cast<uint64_t>(bytes.size()) * 8;
    bytes.push_back(0x80);
    while ((bytes.size() % 64) != 56) {
        bytes.push_back(0);
    }
    for (int i = 7; i >= 0; --i) {
        bytes.push_back(static_cast<uint8_t>(bits >> (8 * i)));
    }

    for (size_t off = 0; off < bytes.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(bytes[off + 4 * i]) << 24) |
                   (static_cast<uint32_t>(bytes[off + 4 * i + 1]) << 16) |
                   (static_cast<uint32_t>(bytes[off + 4 * i + 2]) << 8) |
                   static_cast<uint32_t>(bytes[off + 4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;

            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    for (int i = 0; i < 8; ++i) {
        out[4 * i] = static_cast<uint8_t>(h[i] >> 24);
        out[4 * i + 1] = static_cast<uint8_t>(h[i] >> 16);
        out[4 * i + 2] = static_cast<uint8_t>(h[i] >> 8);
        out[4 * i + 3] = static_cast<uint8_t>(h[i]);
    }
}

inline void append_le_scalar(std::vector<uint8_t>& state, const Fr& scalar) {
    uint8_t be[32];
    fr_to_bytes(scalar, be);
    for (int i = 31; i >= 0; --i) {
        state.push_back(be[i]);
    }
}

// Fiat-Shamir transcript matching rust-verkle's ipa-multipoint transcript domain separation
struct Transcript {
    std::vector<uint8_t> state;

    Transcript() {
        const uint8_t domain_label[] = {'i', 'p', '_', 'n', 'o', '_', 'z', 'k'};
        state.insert(state.end(), domain_label, domain_label + sizeof(domain_label));
    }

    void label(const char* str) {
        while (*str) {
            state.push_back(static_cast<uint8_t>(*str++));
        }
    }

    void point(const char* tag, const PointExtended& p) {
        label(tag);
        uint8_t bytes[32];
        bw_to_bytes({p}, bytes);
        state.insert(state.end(), bytes, bytes + 32);
    }

    void scalar(const char* tag, const Fr& s) {
        label(tag);
        append_le_scalar(state, s);
    }

    Fr challenge(const char* tag) {
        label(tag);
        uint8_t digest[32];
        sha256(state, digest);
        state.clear();

        uint32_t raw[8];
        for (int i = 0; i < 8; ++i) {
            raw[i] = static_cast<uint32_t>(digest[4 * i]) |
                    (static_cast<uint32_t>(digest[4 * i + 1]) << 8) |
                    (static_cast<uint32_t>(digest[4 * i + 2]) << 16) |
                    (static_cast<uint32_t>(digest[4 * i + 3]) << 24);
        }
        Fr result = fr_from_raw(raw);
        scalar(tag, result);
        return result;
    }
};

inline Fr inner_product(const Fr* a, const Fr* b, int n) {
    Fr result = FR_ZERO;
    for (int i = 0; i < n; ++i) {
        result = fr_add(result, fr_mul(a[i], b[i]));
    }
    return result;
}

inline PointExtended multiexp(const Fr* scalars, const PointExtended* points, int n) {
    PointExtended result = point_identity();
    for (int i = 0; i < n; ++i) {
        if (!fr_is_zero(scalars[i])) {
            result = point_add(result, scalar_mul(points[i], scalars[i]));
        }
    }
    return result;
}

} // namespace ipa_detail

// Logarithmic IPA opening proof (8 L points, 8 R points, 1 final scalar)
struct IpaOpeningProof {
    std::array<PointExtended, IPA_ROUNDS> L{};
    std::array<PointExtended, IPA_ROUNDS> R{};
    Fr final_scalar = FR_ZERO;
};

// Wire format compatible with ipa-multipoint's IPAProof::to_bytes():
// - 8 compressed L points (256 bytes)
// - 8 compressed R points (256 bytes)
// - 1 canonical little-endian scalar (32 bytes)
// Total = 544 bytes
inline void ipa_proof_to_bytes(const IpaOpeningProof& proof, uint8_t out[IPA_PROOF_BYTES]) {
    size_t offset = 0;
    for (const PointExtended& point : proof.L) {
        bw_to_bytes({point}, out + offset);
        offset += 32;
    }
    for (const PointExtended& point : proof.R) {
        bw_to_bytes({point}, out + offset);
        offset += 32;
    }
    uint8_t scalar_be[32];
    fr_to_bytes(proof.final_scalar, scalar_be);
    for (int i = 0; i < 32; ++i) {
        out[offset + i] = scalar_be[31 - i];
    }
}

// Strict wire decoder validating all subgroup and canonical encoding constraints
inline bool ipa_proof_from_bytes_strict(const uint8_t* in, size_t length, IpaOpeningProof& proof) {
    if (in == nullptr || length != IPA_PROOF_BYTES) {
        return false;
    }

    IpaOpeningProof candidate;
    size_t offset = 0;

    for (PointExtended& point : candidate.L) {
        BanderwagonElement decoded;
        if (!bw_from_bytes_strict(in + offset, decoded)) return false;
        point = decoded.point;
        offset += 32;
    }

    for (PointExtended& point : candidate.R) {
        BanderwagonElement decoded;
        if (!bw_from_bytes_strict(in + offset, decoded)) return false;
        point = decoded.point;
        offset += 32;
    }

    uint8_t scalar_be[32];
    for (int i = 0; i < 32; ++i) {
        scalar_be[i] = in[offset + 31 - i];
    }
    if (!fr_from_bytes_strict(scalar_be, candidate.final_scalar)) {
        return false;
    }

    proof = candidate;
    return true;
}

// Compute powers of the evaluation point: [1, z, z^2, ..., z^(IPA_WIDTH-1)]
inline void ipa_powers(const Fr& z, Fr out[IPA_WIDTH]) {
    out[0] = FR_ONE;
    for (int i = 1; i < IPA_WIDTH; ++i) {
        out[i] = fr_mul(out[i - 1], z);
    }
}

// Evaluate polynomial at point z: y = <values, [1, z, z^2, ...]>
inline Fr ipa_evaluate(const Fr values[IPA_WIDTH], const Fr& z) {
    Fr b[IPA_WIDTH];
    ipa_powers(z, b);
    return ipa_detail::inner_product(values, b, IPA_WIDTH);
}

// Compute Pedersen commitment for the 256-element vector
inline PointExtended ipa_commit(const Fr values[IPA_WIDTH], const crs::CRSPoints& crs) {
    return msm_compute(values, crs.x, crs.y, IPA_WIDTH);
}

// Prover: generates a logarithmic IPA opening proof for <values, b> = evaluation
inline bool ipa_prove(
    const Fr values[IPA_WIDTH],
    const Fr& z,
    IpaOpeningProof& proof,
    Fr& evaluation,
    const crs::CRSPoints& crs)
{
    Fr a[IPA_WIDTH];
    Fr b[IPA_WIDTH];
    PointExtended g[IPA_WIDTH];

    for (int i = 0; i < IPA_WIDTH; ++i) {
        a[i] = values[i];
        g[i] = point_from_affine({crs.x[i], crs.y[i]});
    }

    ipa_powers(z, b);
    evaluation = ipa_detail::inner_product(a, b, IPA_WIDTH);

    const PointExtended commitment = ipa_commit(values, crs);

    // Initialize Fiat-Shamir transcript
    ipa_detail::Transcript tr;
    tr.label("ipa");
    tr.point("C", commitment);
    tr.scalar("input point", z);
    tr.scalar("output point", evaluation);

    const Fr w = tr.challenge("w");
    const PointExtended q = scalar_mul(point_from_affine({crs.q_x, crs.q_y}), w);

    int n = IPA_WIDTH;

    // Halving rounds: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    for (int round = 0; round < IPA_ROUNDS; ++round) {
        const int h = n / 2;

        const Fr z_l = ipa_detail::inner_product(a + h, b, h);
        const Fr z_r = ipa_detail::inner_product(a, b + h, h);

        Fr ls[129], rs[129];
        PointExtended lp[129], rp[129];

        for (int i = 0; i < h; ++i) {
            ls[i] = a[h + i];
            lp[i] = g[i];
            rs[i] = a[i];
            rp[i] = g[h + i];
        }
        ls[h] = z_l;
        lp[h] = q;
        rs[h] = z_r;
        rp[h] = q;

        proof.L[round] = ipa_detail::multiexp(ls, lp, h + 1);
        proof.R[round] = ipa_detail::multiexp(rs, rp, h + 1);

        tr.point("L", proof.L[round]);
        tr.point("R", proof.R[round]);

        const Fr x = tr.challenge("x");
        if (fr_is_zero(x)) return false;

        const Fr xi = fr_inv(x);

        for (int i = 0; i < h; ++i) {
            a[i] = fr_add(a[i], fr_mul(x, a[h + i]));
            b[i] = fr_add(b[i], fr_mul(xi, b[h + i]));
            g[i] = point_add(g[i], scalar_mul(g[h + i], xi));
        }

        n = h;
    }

    proof.final_scalar = a[0];
    return true;
}

// Verifier: verifies an IPA opening proof against commitment C at point z
inline bool ipa_verify(
    const PointExtended& commitment,
    const Fr& z,
    const Fr& evaluation,
    const IpaOpeningProof& proof,
    const crs::CRSPoints& crs)
{
    Fr b[IPA_WIDTH];
    PointExtended g[IPA_WIDTH];

    ipa_powers(z, b);
    for (int i = 0; i < IPA_WIDTH; ++i) {
        g[i] = point_from_affine({crs.x[i], crs.y[i]});
    }

    // Replay Fiat-Shamir transcript
    ipa_detail::Transcript tr;
    tr.label("ipa");
    tr.point("C", commitment);
    tr.scalar("input point", z);
    tr.scalar("output point", evaluation);

    const Fr w = tr.challenge("w");
    const PointExtended q = scalar_mul(point_from_affine({crs.q_x, crs.q_y}), w);

    PointExtended p = point_add(commitment, scalar_mul(q, evaluation));
    int n = IPA_WIDTH;

    for (int round = 0; round < IPA_ROUNDS; ++round) {
        tr.point("L", proof.L[round]);
        tr.point("R", proof.R[round]);

        const Fr x = tr.challenge("x");
        if (fr_is_zero(x)) return false;

        const Fr xi = fr_inv(x);

        p = point_add(p, point_add(scalar_mul(proof.L[round], x), scalar_mul(proof.R[round], xi)));

        const int h = n / 2;
        for (int i = 0; i < h; ++i) {
            g[i] = point_add(g[i], scalar_mul(g[h + i], xi));
            b[i] = fr_add(b[i], fr_mul(xi, b[h + i]));
        }

        n = h;
    }

    const PointExtended expected = point_add(
        scalar_mul(g[0], proof.final_scalar),
        scalar_mul(q, fr_mul(proof.final_scalar, b[0]))
    );

    return bw_eq({p}, {expected});
}
