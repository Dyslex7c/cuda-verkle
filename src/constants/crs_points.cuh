#pragma once

#include "../field/fp.cuh"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "crs_data.inc"

namespace crs {

// mumber of basis points in the CRS
static constexpr int CRS_SIZE = 256;

// Host-side storage for CRS points in SoA format, these are loaded from the precomputed data and stored in Montgomery form
struct CRSPoints {
    Fp x[CRS_SIZE]; // x-coordinates of G_0..G_255
    Fp y[CRS_SIZE]; // y-coordinates of G_0..G_255
    Fp q_x; // Q point x-coordinate
    Fp q_y; // Q point y-coordinate
    bool loaded;
};

// parse a hex character to integer
inline int hex_char_to_int(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

// parse 64 hex characters (32 bytes in little-endian order) into an Fp
inline Fp parse_hex_coordinate(const char* hex) {
    uint32_t limbs[8] = {0};
    for (int i = 0; i < 8; i++) {
        uint32_t limb = 0;
        for (int j = 0; j < 4; j++) {
            int offset = (i * 4 + j) * 2;
            uint32_t byte_val = (hex_char_to_int(hex[offset]) << 4) | 
                                 hex_char_to_int(hex[offset + 1]);
            limb |= (byte_val << (j * 8));
        }
        limbs[i] = limb;
    }
    return fp_from_raw(limbs);
}

// parse a 128-hex-character string into x and y Fp coordinates
inline void parse_hex_point(const char* hex, Fp& out_x, Fp& out_y) {
    out_x = parse_hex_coordinate(hex);
    out_y = parse_hex_coordinate(hex + 64);
}

// loads all CRS points from the precomputed hex array into the CRSPoints struct
inline void load_crs(CRSPoints& points) {
    for (int i = 0; i < CRS_SIZE; i++) {
        parse_hex_point(CRS_HEX[i], points.x[i], points.y[i]);
    }
    // load Q point (index 256)
    parse_hex_point(CRS_HEX[CRS_SIZE], points.q_x, points.q_y);
    points.loaded = true;
}

} // namespace crs
