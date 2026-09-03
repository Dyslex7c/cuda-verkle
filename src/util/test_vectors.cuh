#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include <cctype>
#include <cstdlib>
#include <map>
#include <sstream>

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

// Minimal JSON reader for the repository's deterministic Rust test vectors. It
// intentionally supports only JSON data types and keeps production code free of
// a JSON dependency; it is test-only infrastructure.
struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    double number = 0;
    std::string string;
    std::vector<JsonValue> array;
    std::map<std::string, JsonValue> object;

    const JsonValue& at(const std::string& key) const {
        const auto found = object.find(key);
        if (type != Type::Object || found == object.end()) {
            throw std::runtime_error("Missing JSON key: " + key);
        }
        return found->second;
    }

    const std::string& as_string() const {
        if (type != Type::String) throw std::runtime_error("Expected JSON string");
        return string;
    }

    size_t as_size() const {
        if (type != Type::Number || number < 0 || number != static_cast<size_t>(number)) {
            throw std::runtime_error("Expected non-negative JSON integer");
        }
        return static_cast<size_t>(number);
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& input) : input_(input) {}

    JsonValue parse() {
        JsonValue value = parse_value();
        skip_whitespace();
        if (position_ != input_.size()) fail("trailing data");
        return value;
    }

private:
    const std::string& input_;
    size_t position_ = 0;

    [[noreturn]] void fail(const char* message) const {
        throw std::runtime_error(std::string("Invalid JSON at offset ") +
                                 std::to_string(position_) + ": " + message);
    }

    void skip_whitespace() {
        while (position_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[position_]))) ++position_;
    }

    char take() {
        if (position_ == input_.size()) fail("unexpected end of input");
        return input_[position_++];
    }

    void expect(char expected) {
        if (take() != expected) fail("unexpected character");
    }

    bool consume(const char* text) {
        const size_t length = std::strlen(text);
        if (input_.compare(position_, length, text) != 0) return false;
        position_ += length;
        return true;
    }

    JsonValue parse_value() {
        skip_whitespace();
        if (position_ == input_.size()) fail("expected value");
        const char current = input_[position_];
        if (current == '{') return parse_object();
        if (current == '[') return parse_array();
        if (current == '"') return JsonValue{JsonValue::Type::String, false, 0, parse_string()};
        if (current == '-' || std::isdigit(static_cast<unsigned char>(current))) return parse_number();
        if (consume("true")) return JsonValue{JsonValue::Type::Bool, true};
        if (consume("false")) return JsonValue{JsonValue::Type::Bool, false};
        if (consume("null")) return JsonValue{};
        fail("unknown value");
    }

    JsonValue parse_object() {
        JsonValue value;
        value.type = JsonValue::Type::Object;
        expect('{');
        skip_whitespace();
        if (position_ < input_.size() && input_[position_] == '}') { ++position_; return value; }
        while (true) {
            skip_whitespace();
            if (position_ == input_.size() || input_[position_] != '"') fail("expected object key");
            const std::string key = parse_string();
            skip_whitespace();
            expect(':');
            value.object.emplace(key, parse_value());
            skip_whitespace();
            const char separator = take();
            if (separator == '}') return value;
            if (separator != ',') fail("expected object separator");
        }
    }

    JsonValue parse_array() {
        JsonValue value;
        value.type = JsonValue::Type::Array;
        expect('[');
        skip_whitespace();
        if (position_ < input_.size() && input_[position_] == ']') { ++position_; return value; }
        while (true) {
            value.array.push_back(parse_value());
            skip_whitespace();
            const char separator = take();
            if (separator == ']') return value;
            if (separator != ',') fail("expected array separator");
        }
    }

    std::string parse_string() {
        expect('"');
        std::string value;
        while (true) {
            const char current = take();
            if (current == '"') return value;
            if (static_cast<unsigned char>(current) < 0x20) fail("control character in string");
            if (current != '\\') { value += current; continue; }
            const char escaped = take();
            switch (escaped) {
                case '"': value += '"'; break;
                case '\\': value += '\\'; break;
                case '/': value += '/'; break;
                case 'b': value += '\b'; break;
                case 'f': value += '\f'; break;
                case 'n': value += '\n'; break;
                case 'r': value += '\r'; break;
                case 't': value += '\t'; break;
                default: fail("unsupported string escape");
            }
        }
    }

    JsonValue parse_number() {
        const size_t start = position_;
        if (input_[position_] == '-') ++position_;
        if (position_ == input_.size() || !std::isdigit(static_cast<unsigned char>(input_[position_]))) fail("invalid number");
        while (position_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[position_]))) ++position_;
        if (position_ < input_.size() && input_[position_] == '.') {
            ++position_;
            while (position_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[position_]))) ++position_;
        }
        if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-')) ++position_;
            while (position_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[position_]))) ++position_;
        }
        JsonValue value;
        value.type = JsonValue::Type::Number;
        value.number = std::strtod(input_.c_str() + start, nullptr);
        return value;
    }
};

inline JsonValue read_json(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) throw std::runtime_error("Could not open JSON file: " + filepath);
    std::ostringstream contents;
    contents << file.rdbuf();
    return JsonParser(contents.str()).parse();
}

inline std::string vector_path(const std::string& filename) {
#ifdef CUDA_VERKLE_SOURCE_DIR
    return std::string(CUDA_VERKLE_SOURCE_DIR) + "/test_vectors/" + filename;
#else
    return "test_vectors/" + filename;
#endif
}

} // namespace test_util
} // namespace cuda_verkle
