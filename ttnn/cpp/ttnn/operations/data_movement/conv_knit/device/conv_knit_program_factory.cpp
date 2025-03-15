// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hostdevcommon/kernel_structs.h"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/buffer_constants.hpp"
#include "tt-metalium/circular_buffer_types.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/logger.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/math.hpp"
#include <cstdint>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_align.hpp>
#include "conv_knit_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

std::vector<uint32_t> split_work(uint32_t num_inputs_per_core_unpadded, uint32_t num_cores = 5) {
    uint32_t base = num_inputs_per_core_unpadded / num_cores;
    uint32_t remainder = num_inputs_per_core_unpadded % num_cores;

    std::vector<uint32_t> work_distribution(num_cores, base);

    for (uint32_t i = 0; i < remainder; ++i) {
        work_distribution[i] += 1;
    }

    return work_distribution;
}

operation::ProgramWithCallbacks conv_knit_multi_core(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    uint32_t kernel_height,
    uint32_t num_output_channels,
    uint32_t input_width,
    uint32_t num_input_channels) {
    tt::tt_metal::Program program{};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    log_info(
        tt::LogOp, "Input data format is: {} Output data format is: {}", input_cb_data_format, output_cb_data_format);

    ShardSpec input_shard_spec = input_tensor.shard_spec().value();
    TensorMemoryLayout tensor_memory_layout = input_tensor.memory_config().memory_layout;  // needs to be height-sharded
    bool rm_orientation = input_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    CoreRangeSet all_cores = input_tensor.shard_spec().value().grid;
    uint32_t num_cores = all_cores.num_cores();
    std::vector<CoreCoord> cores = corerange_to_cores(all_cores, std::nullopt, rm_orientation);

    log_info(tt::LogOp, "Num input channels is: {}", num_input_channels);

    uint32_t input_unit_size = input_shard_spec.shape[1] * input_tensor.element_size();
    uint32_t num_inputs_per_core_unpadded = input_shard_spec.shape[0];
    uint32_t num_inputs_height = input_tensor.volume() / input_tensor.get_padded_shape()[-1];
    log_info(
        tt::LogOp,
        "Input unit size is {} num_inputs_per_core_unpadded is {}",
        input_unit_size,
        num_inputs_per_core_unpadded);

    ShardSpec output_shard_spec = output_tensor.shard_spec().value();
    uint32_t num_outputs_per_core_unpadded = output_shard_spec.shape[0];
    uint32_t num_outputs_height = output_tensor.volume() / output_tensor.get_padded_shape()[-1];
    log_info(
        tt::LogOp,
        "Output unit size is {} num_outputs_per_core_unpadded is {}",
        input_unit_size,
        num_outputs_per_core_unpadded);

    uint32_t src_cb_index = tt::CBIndex::c_0;
    uint32_t out_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig src_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_inputs_per_core_unpadded * input_unit_size, {{src_cb_index, input_cb_data_format}})
            .set_page_size(src_cb_index, input_unit_size)
            .set_globally_allocated_address(*input_tensor.buffer());
    auto cb_src = tt::tt_metal::CreateCircularBuffer(program, all_cores, src_cb_config);
    // todo: pp, support different page sizes for input and output

    uint32_t output_unit_size = input_unit_size;
    tt::tt_metal::CircularBufferConfig out_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_outputs_per_core_unpadded * output_unit_size, {{out_cb_index, output_cb_data_format}})
            .set_page_size(out_cb_index, output_unit_size)
            .set_globally_allocated_address(*output_tensor.buffer());

    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, out_cb_config);
    log_info(tt::LogOp, "Cb1 total size {}: CB2 total size {}", src_cb_config.total_size(), out_cb_config.total_size());

    std::vector<uint32_t> num_sticks_per_riscv = split_work(num_inputs_per_core_unpadded, 2);  // NC and BR for now
    log_info(tt::LogOp, "Num sticks per riscv: NC:{} BR:{}", num_sticks_per_riscv[0], num_sticks_per_riscv[1]);

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();

    uint32_t current_riscv_stick_starting_index = 0;

    std::vector<uint32_t> kernel_compile_time_args = {
        (std::uint32_t)src_cb_index,
        (std::uint32_t)out_cb_index,
        input_unit_size,
        num_input_channels,
        input_width,
        num_output_channels,
        num_inputs_per_core_unpadded,
        num_sticks_per_riscv[0],
        current_riscv_stick_starting_index};
    tt::tt_metal::KernelHandle nc_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/conv_knit/device/kernels/dataflow/"
        "reader_writer_conv_knit_move_sticks_height_sharded.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(kernel_compile_time_args));

    current_riscv_stick_starting_index += num_sticks_per_riscv[0];
    kernel_compile_time_args[7] = num_sticks_per_riscv[1];
    kernel_compile_time_args[8] = current_riscv_stick_starting_index;
    tt::tt_metal::KernelHandle br_kernel_handle = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/conv_knit/device/kernels/dataflow/"
        "reader_writer_conv_knit_move_sticks_height_sharded.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(kernel_compile_time_args));

    current_riscv_stick_starting_index += num_sticks_per_riscv[1];

    auto override_runtime_arguments_callback = [cb_src, cb_output](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, cb_src, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::data_movement::detail
