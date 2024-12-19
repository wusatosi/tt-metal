
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "tt_metal/common/assert.hpp"

#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

class float16 {
private:
    uint16_t uint16_data;

public:
    static constexpr size_t SIZEOF = 2;

    float16() = default;

    // Constructor to create from a float
    float16(float float_num) {
        static_assert(sizeof(float_num) == 4, "float must have size 4");

        // Convert float to uint32_t representation
        uint32_t bits = *reinterpret_cast<uint32_t*>(&float_num);

        // Extract the sign, exponent, and mantissa
        uint32_t sign = (bits >> 31) & 0x1;
        uint32_t exponent = (bits >> 23) & 0xFF;
        uint32_t mantissa = bits & 0x7FFFFF;

        // Handle denormalized numbers
        if (exponent == 0) {
            // For subnormal numbers, the exponent is stored as 0
            exponent = 0;
        } else if (exponent == 255) {
            // Handle infinity and NaN
            exponent = 31;  // Max exponent for half-precision
            mantissa = 0;
        } else {
            // For normalized numbers, adjust exponent to fit in 5 bits
            exponent -= 127;  // Bias for single-precision
            if (exponent > 30) exponent = 31;  // Cap at max exponent
            if (exponent < 0) exponent = 0;   // Handle subnormal (denorm) case
        }

        // Truncate mantissa to fit in 10 bits
        mantissa >>= 13;  // Only keep the upper 10 bits

        // Combine into 16-bit representation
        uint16_data = (sign << 15) | (exponent << 10) | mantissa;
    }

    // Constructor from raw 16-bit packed representation
    float16(uint16_t uint16_data_) : uint16_data(uint16_data_) {}

    // Convert back to float
    float to_float() const {
        uint32_t sign = (uint16_data >> 15) & 0x1;
        uint32_t exponent = (uint16_data >> 10) & 0x1F;
        uint32_t mantissa = uint16_data & 0x3FF;

        // Handle special cases: zero, subnormal, infinity, NaN
        if (exponent == 0) {
            if (mantissa == 0) {
                return 0.0f;  // Zero
            } else {
                // Subnormal numbers
                exponent = 1;
            }
        } else if (exponent == 31) {
            // Handle infinity or NaN
            if (mantissa == 0) {
                return std::numeric_limits<float>::infinity();  // Infinity
            } else {
                return std::numeric_limits<float>::quiet_NaN();  // NaN
            }
        }

        // Normalize the exponent and mantissa for the float representation
        exponent += 127;  // Bias for single-precision
        mantissa <<= 13;  // Rebuild the full mantissa

        uint32_t bits = (sign << 31) | (exponent << 23) | mantissa;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    uint16_t to_packed() const { return uint16_data; }
    uint16_t to_uint16() const { return uint16_data; }

    bool operator==(const float16 rhs) const { return uint16_data == rhs.uint16_data; }
    bool operator!=(const float16 rhs) const { return not(*this == rhs); }

    float16 operator*(const float16 rhs) const { return float16(this->to_float() * rhs.to_float()); }

    void print() const {
        std::cout << "float16: " << to_float() << std::endl;
    }
};

inline std::pair<float16, float16> unpack_two_float16_from_uint32(uint32_t uint32_data) {
    std::pair<float16, float16> two_floats;

    two_floats.first = float16(uint16_t(uint32_data & 0xffff));  // lower 16
    two_floats.second = float16(uint16_t(uint32_data >> 16));    // upper 16

    return two_floats;
}


inline std::vector<std::uint32_t> create_random_vector_of_float16(
    uint32_t num_bytes, int rand_max_float, int seed, float offset = 0.0f) {
    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, rand_max_float), std::mt19937(seed));

    std::vector<std::uint32_t> vec(num_bytes / sizeof(std::uint32_t), 0);
    for (int i = 0; i < vec.size(); i++) {
        float num_1_float = rand_float() + offset;
        float num_2_float = rand_float() + offset;

        float16 num_1_float16 = float16(num_1_float);
        float16 num_2_float16 = float16(num_2_float);


        // pack 2 uint16 into uint32
        vec.at(i) = (uint32_t)num_1_float16.to_uint16() | ((uint32_t)num_2_float16.to_uint16() << 16);
    }

    return vec;
}

inline float16 float16_identity_transform(const float16& input) { return input; }

inline std::vector<float16> unpack_uint32_vec_into_float16_vec(
    const std::vector<std::uint32_t>& data,
    std::function<float16(const float16&)> transform = float16_identity_transform) {
    std::vector<float16> result;
    for (auto i = 0; i < data.size(); i++) {
        auto unpacked = unpack_two_float16_from_uint32(data[i]);
        result.push_back(transform(unpacked.first));
        result.push_back(transform(unpacked.second));
    }
    return result;
}