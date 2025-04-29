// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <bit>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/float32.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "umd/device/types/arch.h"
#include <tt-metalium/utils.hpp>

namespace tt::tt_metal {
class IDevice;

using std::map;
using std::vector;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::binary {
const map<string, string> binary_op_name_to_op_type = {
    {"add", "EltwiseBinaryType::ELWADD"},
    {"sub", "EltwiseBinaryType::ELWSUB"},
    {"mul", "EltwiseBinaryType::ELWMUL"},
    {"add_with_dest_reuse", "EltwiseBinaryType::ELWADD"},
    {"sub_with_dest_reuse", "EltwiseBinaryType::ELWSUB"},
    {"mul_with_dest_reuse", "EltwiseBinaryType::ELWMUL"},
};
const map<string, string> binary_op_name_to_op_kernel = {
    {"add", "add_tiles"},
    {"sub", "sub_tiles"},
    {"mul", "mul_tiles"},
};

struct SingleCoreBinaryConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    size_t input_dram_byte_address = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    CoreCoord core = {};
    std::string binary_op = "";
    bool acc_to_dest = false;
    bool full_init = true;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
};

void set_math_fid_masks(
    uint16_t& srca_fid_mask, uint16_t& srcb_fid_mask, MathFidelity math_fidelity = MathFidelity::HiFi4) {
    auto arch = get_arch_from_string(get_umd_arch_name());
    switch (math_fidelity) {
        case MathFidelity::HiFi4:
        case MathFidelity::HiFi3: {
            break;
        }
        case MathFidelity::HiFi2: {
            srcb_fid_mask = (arch == tt::ARCH::GRAYSKULL) ? 0xFFF8 : 0xFFFE;
            ;
            break;
        }
        case MathFidelity::LoFi: {
            srca_fid_mask = 0xFFF8;
            srcb_fid_mask = (arch == tt::ARCH::GRAYSKULL) ? 0xFFF8 : 0xFFFE;
            break;
        }
        default: {
            TT_THROW("Unsupported MathFidelity={}", math_fidelity);
            break;
        }
    }
}

/// @brief Does Dramx2 --> Reader --> CB --> Binary Compute --> CB --> Writer --> Dram
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool single_core_binary(tt_metal::IDevice* device, const SingleCoreBinaryConfig& test_config) {
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    tt_metal::Program program = tt_metal::CreateProgram();

    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = device, .size = byte_size, .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM};
    auto input0_dram_buffer = CreateBuffer(dram_config);
    uint32_t input0_dram_byte_address = input0_dram_buffer->address();

    auto input1_dram_buffer = CreateBuffer(dram_config);
    uint32_t input1_dram_byte_address = input1_dram_buffer->address();

    auto input2_dram_buffer = CreateBuffer(dram_config);
    uint32_t input2_dram_byte_address = input2_dram_buffer->address();

    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t output_dram_byte_address = output_dram_buffer->address();

    tt_metal::CircularBufferConfig l1_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{0, test_config.l1_input_data_format}})
            .set_page_size(0, test_config.tile_byte_size);
    auto l1_input0_cb = tt_metal::CreateCircularBuffer(program, test_config.core, l1_cb_config);

    tt_metal::CircularBufferConfig l1_input1_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{1, test_config.l1_input_data_format}})
            .set_page_size(1, test_config.tile_byte_size);
    auto l1_input1_cb = tt_metal::CreateCircularBuffer(program, test_config.core, l1_input1_cb_config);

    tt_metal::CircularBufferConfig l2_input1_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{2, test_config.l1_input_data_format}})
            .set_page_size(2, test_config.tile_byte_size);
    auto l1_input2_cb = tt_metal::CreateCircularBuffer(program, test_config.core, l2_input1_cb_config);

    tt_metal::CircularBufferConfig l1_output_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{16, test_config.l1_output_data_format}})
            .set_page_size(16, test_config.tile_byte_size);
    auto l1_output_cb = tt_metal::CreateCircularBuffer(program, test_config.core, l1_output_cb_config);

    vector<uint32_t> compute_kernel_args = {};
    std::map<string, string> defines = {{"ELTWISE_OP_TYPE", binary_op_name_to_op_type.at(test_config.binary_op)}};

    if (test_config.binary_op.find("_with_dest_reuse") != std::string::npos) {
        defines["ELTWISE_DEST_REUSE_TYPE"] = "EltwiseBinaryReuseDestType::DEST_TO_SRCA";
    } else {
        defines["ELTWISE_OP"] = binary_op_name_to_op_kernel.at(test_config.binary_op);
        if (test_config.full_init) {
            defines["FULL_INIT"] = "1";
        }
        if (test_config.acc_to_dest) {
            defines["DST_ACCUM_MODE"] = "1";
            defines["ELTWISE_OP_INIT"] = defines["ELTWISE_OP"] + "_init";
            if (test_config.binary_op == "mul") {
                defines["MUL_TILES_WITH_DST_ACCUM"] = "1";
            }
        }
    }

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        test_config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .defines = defines});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        test_config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto binary_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        test_config.core,
        tt_metal::ComputeConfig{
            .math_fidelity = test_config.math_fidelity, .compile_args = compute_kernel_args, .defines = defines});

    SetRuntimeArgs(program, binary_kernel, test_config.core, {uint32_t(test_config.num_tiles), 1});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> packed_input0 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, byte_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input1 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, byte_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input2 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, byte_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    auto input0 = unpack_vector<bfloat16, uint32_t>(packed_input0);
    auto input1 = unpack_vector<bfloat16, uint32_t>(packed_input1);
    auto input2 = unpack_vector<bfloat16, uint32_t>(packed_input2);

    std::vector<float> temp_golden(input0.size());
    uint16_t srca_fid_mask = 0xFFFF;
    uint16_t srcb_fid_mask = 0xFFFF;
    set_math_fid_masks(srca_fid_mask, srcb_fid_mask, test_config.math_fidelity);
    std::transform(
        input0.begin(),
        input0.end(),
        input1.begin(),
        temp_golden.begin(),
        [&](const bfloat16& lhs, const bfloat16& rhs) {
            if (test_config.binary_op == "add") {
                return (lhs.to_float() + rhs.to_float());
            } else if (test_config.binary_op == "sub") {
                return (lhs.to_float() - rhs.to_float());
            } else if (test_config.binary_op == "mul") {
                return (
                    bfloat16(std::bit_cast<uint32_t>(lhs.to_packed() & srca_fid_mask)).to_float() *
                    bfloat16(std::bit_cast<uint32_t>(rhs.to_packed() & srcb_fid_mask)).to_float());
            } else if (test_config.binary_op.find("with_dest_reuse") != std::string::npos) {
                return lhs.to_float();
            } else {
                TT_THROW("Unsupported binary_op={}", test_config.binary_op);
                return 0.0f;
            }
        });

    std::vector<bfloat16> golden(input0.size());
    std::transform(
        input2.begin(), input2.end(), temp_golden.begin(), golden.begin(), [&](const bfloat16& lhs, const float& rhs) {
            // acc_to_dest accumulates dest value with binary output, for all binary operations
            if (test_config.acc_to_dest || test_config.binary_op == "add_with_dest_reuse") {
                return (lhs.to_float() + rhs);
            } else if (test_config.binary_op == "sub_with_dest_reuse") {
                return (lhs.to_float() - rhs);
            } else if (test_config.binary_op == "mul_with_dest_reuse") {
                return (
                    bfloat16(std::bit_cast<uint32_t>(lhs.to_packed() & srca_fid_mask)).to_float() *
                    bfloat16(std::bit_cast<uint32_t>(bfloat16(rhs).to_packed() & srcb_fid_mask)).to_float());
            } else {
                return rhs;
            }
        });
    auto packed_golden = pack_vector<uint32_t, bfloat16>(golden);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::detail::WriteToBuffer(input0_dram_buffer, packed_input0);
    tt_metal::detail::WriteToBuffer(input1_dram_buffer, packed_input1);
    tt_metal::detail::WriteToBuffer(input2_dram_buffer, packed_input2);

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        test_config.core,
        {
            input0_dram_byte_address,
            0,  // dram bank id
            input1_dram_byte_address,
            0,  // dram bank id
            static_cast<std::uint32_t>(test_config.num_tiles),
            input2_dram_byte_address,
            0,  // dram bank id
        });
    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        test_config.core,
        {
            output_dram_byte_address,
            0,  // dram bank id
            static_cast<std::uint32_t>(test_config.num_tiles),
        });

    tt_metal::detail::LaunchProgram(device, program);

    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);

    pass &= is_close_packed_vectors<bfloat16, uint32_t>(
        dest_buffer_data, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.0155f); });
    return pass;
}
}  // namespace unit_tests::compute::binary

using TensixBinaryComputeSingleCoreTestParameters = std::tuple<MathFidelity, std::string, std::int32_t, bool>;
using TensixBinaryComputeSingleCoreDestAccTestParameters = std::tuple<MathFidelity, std::string>;

namespace {
std::string ConvertToCamelCase(std::string_view input) {
    std::string result;
    result.reserve(input.size());
    bool capitalize = true;

    for (auto ch : input) {
        if (ch == '_') {
            capitalize = true;
        } else {
            result += capitalize ? std::toupper(ch) : ch;
            capitalize = false;
        }
    }

    return result;
}

auto getFidelityName = [](MathFidelity fidelity) -> std::string {
    static const std::string fidelityNames[] = {
        "LoFi",   // Fidelity::LoFi
        "",       // Fidelity::Invalid
        "HiFi2",  // Fidelity::HiFi2
        "HiFi3",  // Fidelity::HiFi3
        "HiFi4"   // Fidelity::HiFi4
    };
    assert(fidelity >= MathFidelity::LoFi && fidelity <= MathFidelity::HiFi4);
    return fidelityNames[static_cast<int>(fidelity)];
};

std::string TensixBinaryComputeSingleCoreTestNameGenerator(
    const ::testing::TestParamInfo<TensixBinaryComputeSingleCoreTestParameters>& configuration) {
    auto [math_fidelity, binary_op, number_of_tiles, is_full_init] = configuration.param;
    std::string math_fidelity_name = getFidelityName(math_fidelity);
    binary_op[0] = std::toupper(binary_op[0]);
    std::string result =
        ConvertToCamelCase(binary_op) + "_" + math_fidelity_name + "_" + std::to_string(number_of_tiles) + "Tiles";
    if (is_full_init) {
        result += "_FullInit";
    }
    return result;
}

std::string TensixBinaryComputeSingleCoreDestAccTestNameGenerator(
    const ::testing::TestParamInfo<TensixBinaryComputeSingleCoreDestAccTestParameters>& configuration) {
    auto [math_fidelity, binary_op] = configuration.param;
    std::string math_fidelity_name = getFidelityName(math_fidelity);
    binary_op[0] = std::toupper(binary_op[0]);
    std::string result = ConvertToCamelCase(binary_op) + "_" + math_fidelity_name + "_4Tiles";
    return result;
}
};  // namespace

class TensixBinaryComputeSingleCoreFixture
    : public DeviceFixture,
      public testing::WithParamInterface<TensixBinaryComputeSingleCoreTestParameters> {};

TEST_P(TensixBinaryComputeSingleCoreFixture, AllOpsTest) {
    const auto [math_fidelity, binary_op, number_of_tiles, is_full_init] = GetParam();
    unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
        .num_tiles = number_of_tiles,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .core = CoreCoord(0, 0),
        .binary_op = binary_op,
        .full_init = is_full_init,
        .math_fidelity = math_fidelity,
    };
    ASSERT_EQ(num_devices_, devices_.size());
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_[id], test_config));
    }
};

INSTANTIATE_TEST_CASE_P(
    AllTensixBinaryOps,
    TensixBinaryComputeSingleCoreFixture,
    testing::Combine(
        testing::Values(MathFidelity::LoFi, MathFidelity::HiFi2, MathFidelity::HiFi3, MathFidelity::HiFi4),
        testing::Values("add", "sub", "mul", "add_with_dest_reuse", "sub_with_dest_reuse", "mul_with_dest_reuse"),
        testing::Values(1, 2, 4),
        testing::Values(true, false)),
    TensixBinaryComputeSingleCoreTestNameGenerator);

class TensixBinaryComputeSingleCoreMultiTileDestAccFixture
    : public DeviceFixture,
      public testing::WithParamInterface<TensixBinaryComputeSingleCoreDestAccTestParameters> {};

TEST_P(TensixBinaryComputeSingleCoreMultiTileDestAccFixture, AllOpsTest) {
    if (arch_ == tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }
    const auto [math_fidelity, binary_op] = GetParam();
    unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
        .num_tiles = 4,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .core = CoreCoord(0, 0),
        .binary_op = binary_op,
        .acc_to_dest = true,
        .math_fidelity = math_fidelity,
    };
    ASSERT_EQ(num_devices_, devices_.size());
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(devices_[id], test_config));
    }
};

INSTANTIATE_TEST_CASE_P(
    AllTensixBinaryOps,
    TensixBinaryComputeSingleCoreMultiTileDestAccFixture,
    testing::Combine(
        testing::Values(MathFidelity::LoFi, MathFidelity::HiFi2, MathFidelity::HiFi3, MathFidelity::HiFi4),
        testing::Values("add", "sub", "mul")),
    TensixBinaryComputeSingleCoreDestAccTestNameGenerator);

}  // namespace tt::tt_metal
