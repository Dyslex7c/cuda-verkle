#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "../src/proof/ipa.cuh"
#include "../src/util/test_vectors.cuh"

static int passed = 0;
static int failed = 0;

#define CHECK(condition, message) do { \
    if (condition) { \
        ++passed; \
        std::printf("  PASS: %s\n", message); \
    } else { \
        ++failed; \
        std::printf("  FAIL: %s\n", message); \
    } \
} while (0)

static bool decode_hex_32(const std::string& hex, uint8_t out[32]) {
    if (hex.size() != 64) return false;
    try {
        for (int i = 0; i < 32; ++i) {
            out[i] = static_cast<uint8_t>(
                (cuda_verkle::test_util::hex_char_to_int(hex[2 * i]) << 4) |
                cuda_verkle::test_util::hex_char_to_int(hex[2 * i + 1])
            );
        }
        return true;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

static bool decode_scalar_be(const std::string& hex, Fr& out) {
    uint8_t bytes[32];
    return decode_hex_32(hex, bytes) && fr_from_bytes_strict(bytes, out);
}

void test_ipa_prove_and_verify(const crs::CRSPoints& crs, Fr values[IPA_WIDTH], const Fr& z, IpaOpeningProof& proof, Fr& y, PointExtended& commitment) {
    for (int i = 0; i < IPA_WIDTH; ++i) {
        values[i] = fr_from_u64(static_cast<uint64_t>(i + 1) * 17);
    }

    CHECK(ipa_prove(values, z, proof, y, crs), "prover creates a logarithmic IPA opening proof");

    commitment = ipa_commit(values, crs);
    CHECK(ipa_verify(commitment, z, y, proof, crs), "verifier accepts valid committed-vector opening");
}

void test_ipa_wire_encoding(const crs::CRSPoints& crs, const PointExtended& commitment, const Fr& z, const Fr& y, const IpaOpeningProof& proof) {
    uint8_t encoded[IPA_PROOF_BYTES];
    ipa_proof_to_bytes(proof, encoded);

    IpaOpeningProof decoded;
    CHECK(
        ipa_proof_from_bytes_strict(encoded, sizeof(encoded), decoded) &&
        ipa_verify(commitment, z, y, decoded, crs),
        "strict proof decoder round-trips a valid proof"
    );

    CHECK(!ipa_proof_from_bytes_strict(nullptr, IPA_PROOF_BYTES, decoded), "strict proof decoder rejects a null input buffer");
    CHECK(!ipa_proof_from_bytes_strict(encoded, sizeof(encoded) - 1, decoded), "strict proof decoder rejects a truncated proof");

    // Corrupt the scalar byte to exceed field modulus
    encoded[sizeof(encoded) - 1] = 0xff;
    CHECK(!ipa_proof_from_bytes_strict(encoded, sizeof(encoded), decoded), "strict proof decoder rejects a non-canonical final scalar");
}

void test_ipa_soundness(const crs::CRSPoints& crs, const PointExtended& commitment, const Fr& z, const Fr& y, const IpaOpeningProof& proof, const Fr values[IPA_WIDTH]) {
    // Tampered evaluation
    Fr bad_y = fr_add(y, fr_one());
    CHECK(!ipa_verify(commitment, z, bad_y, proof, crs), "verifier rejects a tampered evaluation");

    // Tampered round point
    IpaOpeningProof bad_proof = proof;
    bad_proof.L[3] = point_add(bad_proof.L[3], point_from_affine({crs.x[0], crs.y[0]}));
    CHECK(!ipa_verify(commitment, z, y, bad_proof, crs), "verifier rejects a tampered IPA round point");

    // Tampered commitment
    Fr changed_values[IPA_WIDTH];
    for (int i = 0; i < IPA_WIDTH; ++i) {
        changed_values[i] = values[i];
    }
    changed_values[12] = fr_add(changed_values[12], fr_one());
    PointExtended bad_commitment = ipa_commit(changed_values, crs);
    CHECK(!ipa_verify(bad_commitment, z, y, proof, crs), "verifier binds proof to the commitment");
}

void test_ipa_rust_reference_vector(const crs::CRSPoints& crs, const IpaOpeningProof& native_proof) {
    try {
        const auto vector = cuda_verkle::test_util::read_json(
            cuda_verkle::test_util::vector_path("ipa_test_vectors.json")
        );

        const auto& l = vector.at("l").array;
        const auto& r = vector.at("r").array;

        IpaOpeningProof rust_proof;
        uint8_t bytes[32];
        bool loaded = (l.size() == IPA_ROUNDS && r.size() == IPA_ROUNDS);

        for (int i = 0; loaded && i < IPA_ROUNDS; ++i) {
            BanderwagonElement point;
            loaded = decode_hex_32(l[i].as_string(), bytes) && bw_from_bytes_strict(bytes, point);
            if (loaded) rust_proof.L[i] = point.point;

            loaded = decode_hex_32(r[i].as_string(), bytes) && bw_from_bytes_strict(bytes, point);
            if (loaded) rust_proof.R[i] = point.point;
        }

        Fr rust_z, rust_y;
        BanderwagonElement rust_commitment;

        loaded = loaded &&
            decode_scalar_be(vector.at("input_point").as_string(), rust_z) &&
            decode_scalar_be(vector.at("output_point").as_string(), rust_y) &&
            decode_scalar_be(vector.at("final_scalar").as_string(), rust_proof.final_scalar) &&
            decode_hex_32(vector.at("commitment").as_string(), bytes) &&
            bw_from_bytes_strict(bytes, rust_commitment);

        CHECK(loaded, "loads the pinned Rust IPA vector with strict decoders");
        CHECK(loaded && ipa_verify(rust_commitment.point, rust_z, rust_y, rust_proof, crs),
              "verifier accepts the independently generated Rust IPA proof");

        uint8_t rust_bytes[IPA_PROOF_BYTES];
        uint8_t native_bytes[IPA_PROOF_BYTES];
        ipa_proof_to_bytes(rust_proof, rust_bytes);
        ipa_proof_to_bytes(native_proof, native_bytes);

        CHECK(loaded && std::memcmp(rust_bytes, native_bytes, sizeof(native_bytes)) == 0,
              "prover output matches the pinned Rust proof encoding");

    } catch (const std::exception& e) {
        std::printf("  FAIL: Rust IPA vector error: %s\n", e.what());
        failed += 3;
    }
}

int main() {
    std::printf("IPA Opening Proof Tests\n");

    crs::CRSPoints crs;
    crs::load_crs(crs);

    Fr values[IPA_WIDTH];
    const Fr z = fr_from_u64(7);
    IpaOpeningProof proof;
    Fr y;
    PointExtended commitment;

    test_ipa_prove_and_verify(crs, values, z, proof, y, commitment);
    test_ipa_wire_encoding(crs, commitment, z, y, proof);
    test_ipa_soundness(crs, commitment, z, y, proof, values);
    test_ipa_rust_reference_vector(crs, proof);

    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
