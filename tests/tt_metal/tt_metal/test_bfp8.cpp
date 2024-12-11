// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "common/blockfloat_common.hpp"

using namespace tt;

// Helper function to print float bit representation
void print_float_bits(float value) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
    uint32_t sign = (bits >> 31) & 0x1;           // Extract sign bit
    uint32_t exponent = (bits >> 23) & 0xFF;      // Extract exponent (8 bits)
    uint32_t mantissa = (bits & 0x7FFFFF) >> 16;  // Extract top 7 bits of mantissa (truncate to bf16-like)
    log_info(LogTest, "Value: {:10.5f}, Sign: {}, Exponent: {:08b}, Mantissa: {:07b}", value, sign, exponent, mantissa);
}

int main(int argc, char** argv) {
    std::vector<float> floats(16);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<uint32_t> single_row;
    for (float& value : floats) {
        float random_value = dist(gen);
        uint32_t bits = *reinterpret_cast<uint32_t*>(&random_value);
        bits &= 0xFFFF0000;  // Truncate lower 16 bits of the mantissa
        value = *reinterpret_cast<float*>(&bits);
        single_row.push_back(bits);
    }

    log_info(LogTest, "1. BF16 values (bit-level breakdown):");
    for (size_t i = 0; i < floats.size(); ++i) {
        print_float_bits(floats[i]);
    }

    bool is_exp_a = false;
    // get shared exponent
    uint8_t shared_exp = get_max_exp(single_row, is_exp_a);

    log_info(LogTest, "2. Convert BF16 to BFP8_B");
    log_info(LogTest, "Shared Exponent: {:08b}", shared_exp);

    std::vector<uint8_t> bfp8b_mantissa;
    for (auto& value : single_row) {
        uint8_t conv_num = convert_u32_to_bfp<tt::DataFormat::Bfp8_b, false>(value, shared_exp, is_exp_a);
        uint32_t sign = (conv_num >> 7) & 0x1;  // Extract sign bit
        uint32_t mantissa = (conv_num) & 0x7F;  // Extract mantissa (7 bits)
        log_info(
            LogTest,
            "Value: {:10.5f}, Sign: {}, Exponent: {:08b}, Mantissa: {:07b}",
            *reinterpret_cast<float*>(&value),
            sign,
            shared_exp,
            mantissa);
        bfp8b_mantissa.push_back(conv_num);
    }

    log_info(LogTest, "3. Convert from BFP8_B to BF16");
    for (auto& value : bfp8b_mantissa) {
        uint32_t bit_val = convert_bfp_to_u32(tt::DataFormat::Bfp8_b, value, shared_exp, is_exp_a);
        float float_val = *reinterpret_cast<float*>(&bit_val);
        print_float_bits(float_val);
    }

    return 0;
}
