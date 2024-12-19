
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

#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>

class float16 {
private:
    uint16_t uint16_data;

public:
    static constexpr size_t SIZEOF = 2;

    float16() = default;

    // Constructor from float, with rounding
    float16(float float_num) {
        static_assert(sizeof float_num == 4, "float must have size 4");

        uint32_t bits;
        std::memcpy(&bits, &float_num, sizeof(bits));

        uint32_t sign = (bits & 0x80000000) >> 16;      // Extract sign bit
        uint32_t exponent = (bits & 0x7F800000) >> 23;  // Extract exponent
        uint32_t mantissa = bits & 0x007FFFFF;          // Extract mantissa

        if (exponent == 255) {                                   // Handle NaN and infinity
            uint16_data = (sign | 0x7C00 | (mantissa ? 1 : 0));  // Preserve NaN payload minimally
        } else if (exponent > 112) {                             // Normalized number
            exponent -= 127 - 15;
            if (exponent > 0x1F) {  // Overflow to infinity
                uint16_data = sign | 0x7C00;
            } else {
                uint16_data = sign | (exponent << 10) | (mantissa >> 13);
                // Round to nearest, tie to even
                if ((mantissa & 0x1FFF) > 0x1000 || ((mantissa & 0x1FFF) == 0x1000 && (uint16_data & 1))) {
                    uint16_data++;
                }
            }
        } else if (exponent >= 103) {  // Subnormal number
            uint16_data = sign | ((mantissa | 0x800000) >> (126 - exponent));
            // Round to nearest, tie to even
            if ((mantissa & (1 << (126 - exponent - 1))) != 0) {
                uint16_data++;
            }
        } else {  // Underflow to zero
            uint16_data = sign;
        }
    }

    // Constructor from uint16_t (assumes already encoded float16 representation)
    explicit float16(uint16_t uint16_data_) : uint16_data(uint16_data_) {}

    // Convert back to float
    float to_float() const {
        uint32_t sign = (uint16_data & 0x8000) << 16;
        uint32_t exponent = (uint16_data & 0x7C00) >> 10;
        uint32_t mantissa = (uint16_data & 0x03FF);

        uint32_t bits;
        if (exponent == 0) {  // Subnormal or zero
            if (mantissa == 0) {
                bits = sign;  // Zero
            } else {
                // Normalize the subnormal number
                exponent = 1;
                while ((mantissa & 0x0400) == 0) {
                    mantissa <<= 1;
                    exponent--;
                }
                mantissa &= 0x03FF;  // Remove leading 1
                bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
            }
        } else if (exponent == 0x1F) {  // Infinity or NaN
            bits = sign | 0x7F800000 | (mantissa << 13);
        } else {  // Normalized number
            bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }

        float result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    uint16_t to_packed() const { return uint16_data; }

    uint16_t to_uint16() const { return uint16_data; }

    bool operator==(const float16 rhs) const { return uint16_data == rhs.uint16_data; }

    bool operator!=(const float16 rhs) const { return !(*this == rhs); }

    float16 operator*(const float16 rhs) const { return float16(this->to_float() * rhs.to_float()); }
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

        float16 num_1_float16 = float16(5.125f);
        float16 num_2_float16 = float16(5.125f);
        if (i == 0) {
            std::cout << "num_1_float16 = " << num_1_float16.to_float() << std::endl;
            std::cout << "num_2_float16 = " << num_2_float16.to_float() << std::endl;
        }
        // float16 num_1_float16 = float16(num_1_float);
        // float16 num_2_float16 = float16(num_2_float);

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
