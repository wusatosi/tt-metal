// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "non_zero_indices_op.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/cb_utils.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn {

namespace operations::data_movement {

operation::ProgramWithCallbacks non_zero_indices_single_core(
    const Tensor& input, const Tensor& out_num_indices, const Tensor& out_indices) {
    tt::tt_metal::Program program{};
    IDevice* device = input.device();

    uint32_t alignment_base = 32 / input.element_size();
    // we want per core to be aligned to aligment_base per core

    uint32_t aligned_elements = tt::div_up(input.get_padded_shape()[-1], alignment_base) * alignment_base;
    uint32_t actual_elements = input.get_padded_shape()[-1];

    CoreCoord core = {0, 0};

    uint32_t next_cb_index = tt::CBIndex::c_0;
    uint32_t input_cb_index = next_cb_index++;
    uint32_t output_cb_index_0 = next_cb_index++;
    uint32_t output_cb_index_1 = next_cb_index++;

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(DataType::UINT32);
    bool src_is_dram = input.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram_0 = out_num_indices.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram_1 = out_indices.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t page_size = actual_elements * input.element_size();
    uint32_t rounded_page_size = round_up_to_mul32(page_size);
    tt::tt_metal::create_cb(input_cb_index, program, core, rounded_page_size, 2, input_cb_data_format);

    tt::tt_metal::create_cb(output_cb_index_0, program, core, 32, 2, output_cb_data_format);

    uint32_t dst_page_size = actual_elements * 4;
    uint32_t dst_rounded_page_size = round_up_to_mul32(dst_page_size);
    tt::tt_metal::create_cb(output_cb_index_1, program, core, dst_rounded_page_size, 2, output_cb_data_format);

    std::map<string, string> defines;
    defines["NUM_BYTES"] = std::to_string(input.element_size());

    // Create Kernel
    std::vector<uint32_t> compile_time_args = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)output_cb_index_0,
        (std::uint32_t)output_cb_index_1,
        (std::uint32_t)src_is_dram,
        (std::uint32_t)out_is_dram_0,
        (std::uint32_t)out_is_dram_1,
    };

    const std::array run_time_args = {
        (std::uint32_t)input.buffer()->address(),
        (std::uint32_t)out_num_indices.buffer()->address(),
        (std::uint32_t)out_indices.buffer()->address(),
        (std::uint32_t)aligned_elements,
        (std::uint32_t)actual_elements,
        (std::uint32_t)input.element_size()};

    auto kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/non_zero_indices/device/kernels/dataflow/"
        "non_zero_indices_sc_reader.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(compile_time_args, defines));

    tt::tt_metal::SetRuntimeArgs(program, kernel_id, core, run_time_args);

    auto override_runtime_args_callback = [kernel_id, core, page_size](
                                              const void* operation,
                                              const tt::tt_metal::Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto output_0 = output_tensors.at(0);
        auto output_1 = output_tensors.at(1);
        auto input = input_tensors.at(1);
        uint32_t alignment_base = 32 / input.element_size();
        uint32_t aligned_elements = tt::div_up(input.get_padded_shape()[-1], alignment_base) * alignment_base;
        uint32_t actual_elements = input.get_padded_shape()[-1];
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, kernel_id, core);
        runtime_args[0] = input.buffer()->address();
        runtime_args[1] = output_0.buffer()->address();
        runtime_args[2] = output_1.buffer()->address();
        runtime_args[3] = aligned_elements;
        runtime_args[4] = actual_elements;
        runtime_args[5] = input.element_size();
    };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace operations::data_movement

}  // namespace ttnn
