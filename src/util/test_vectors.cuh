#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <iostream>

namespace cuda_verkle {
namespace test_util {

// converts a single hex character to its integer value
inline uint8_t hex_char_to_int(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    throw std::invalid_argument("Invalid hex character");
}

// Loads a hex string into an array of 8x32-bit limbs (little-endian order)
inline void load_hex_to_limbs(const std::string& hex_str, uint32_t limbs[8]) {
    std::string clean_hex = hex_str;
    if (clean_hex.size() >= 2 && (clean_hex.substr(0, 2) == "0x" || clean_hex.substr(0, 2) == "0X")) {
        clean_hex = clean_hex.substr(2);
    }
    
    // Pad with zeros to ensure it represents 256 bits (64 hex characters)
    while (clean_hex.length() < 64) {
        clean_hex = "0" + clean_hex;
    }
    if (clean_hex.length() > 64) {
        throw std::invalid_argument("Hex string too long for 256 bits");
    }

    // Hex string is big-endian, limbs are little-endian
    for (int i = 0; i < 8; ++i) {
        limbs[i] = 0;
        int start = 64 - (i + 1) * 8;
        for (int j = 0; j < 8; ++j) {
            limbs[i] = (limbs[i] << 4) | hex_char_to_int(clean_hex[start + j]);
        }
    }
}

// Loads a curve point from two hex strings representing x and y coordinates
inline void load_point(const std::string& hex_x, const std::string& hex_y, uint32_t x_limbs[8], uint32_t y_limbs[8]) {
    load_hex_to_limbs(hex_x, x_limbs);
    load_hex_to_limbs(hex_y, y_limbs);
}

// file reader for reading test vector lines
inline std::vector<std::string> read_lines(const std::string& filepath) {
    std::vector<std::string> lines;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (!line.empty() && line[0] != '#') { 
            lines.push_back(line);
        }
    }
    return lines;
}

} // namespace test_util
} // namespace cuda_verkle
