// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <cstdint>

#include "deinterleave_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"

namespace ttnn::operations::experimental::deinterleave {

DeinterleaveOperation::ProgramFactory::cached_program_t DeinterleaveOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt::constants;
    using namespace tt::tt_metal::detail;
    using namespace tt::tt_metal;
    using namespace tt;

    log_info(tt::LogOp, "DeinterleaveOperation::ProgramFactory::create; stride_hw: {}", operation_attributes.stride_hw);
    Program program;

    const auto& input = tensor_args.input;

    auto compute_unit_size = [&](const auto& tensor, const auto& data_format) {
        return tensor.get_logical_shape()[-1] * tensor.element_size();
    };

    uint32_t num_units = output.volume() / output.get_logical_shape()[-1];

    auto worker_grid = input.memory_config().shard_spec.value().grid;
    auto num_units_per_core = input.memory_config().shard_spec.value().shape[0];

    uint32_t src_cb_id = CBIndex::c_0;
    auto input_data_format = datatype_to_dataformat_converter(input.get_dtype());
    uint32_t input_unit_size = compute_unit_size(input, input_data_format);
    uint32_t aligned_input_unit_size = round_up_to_mul32(input_unit_size);
    uint32_t src_total_size = input.get_logical_shape()[0] * aligned_input_unit_size;

    tt::tt_metal::CircularBufferConfig src_cb_config =
        tt::tt_metal::CircularBufferConfig(src_total_size, {{src_cb_id, input_data_format}})
            .set_page_size(src_cb_id, aligned_input_unit_size)
            .set_globally_allocated_address(*input.buffer());
    auto src_cb = tt::tt_metal::CreateCircularBuffer(program, worker_grid, src_cb_config);

    uint32_t dst_cb_id = CBIndex::c_1;
    auto output_data_format = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_unit_size = compute_unit_size(output, output_data_format);
    uint32_t aligned_output_unit_size = round_up_to_mul32(output_unit_size);
    uint32_t dst_total_size = output.get_logical_shape()[0] * aligned_output_unit_size;

    tt::tt_metal::CircularBufferConfig dst_cb_config =
        tt::tt_metal::CircularBufferConfig(dst_total_size, {{dst_cb_id, output_data_format}})
            .set_page_size(dst_cb_id, aligned_output_unit_size)
            .set_globally_allocated_address(*output.buffer());
    auto dst_cb = tt::tt_metal::CreateCircularBuffer(program, worker_grid, dst_cb_config);

    std::vector<uint32_t> reader_compile_time_args, writer_compile_time_args;

    TT_FATAL(input_unit_size == output_unit_size, "Deinterleave: input and output unit size must be equal");
    auto height = input.get_logical_shape()[1];
    auto width = input.get_logical_shape()[2];
    auto stick_size = input_unit_size;
    reader_compile_time_args = {
        (uint32_t)src_cb_id,
        (uint32_t)dst_cb_id,
        (uint32_t)width,
        (uint32_t)height,
        (uint32_t)stick_size,
        (uint32_t)1  // AB
    };

    writer_compile_time_args = {
        (uint32_t)src_cb_id,
        (uint32_t)dst_cb_id,
        (uint32_t)width,
        (uint32_t)height,
        (uint32_t)stick_size,
        (uint32_t)0  // CD};
    };

    auto read_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deinterleave/device/kernels/deinterleave_kernel_rm.cpp",
        worker_grid,
        ReaderDataMovementConfig(reader_compile_time_args, {}));

    auto write_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deinterleave/device/kernels/deinterleave_kernel_rm.cpp",
        worker_grid,
        WriterDataMovementConfig(writer_compile_time_args, {}));

    // uint32_t start_id = 0;
    // uint32_t num_cores_group_1 = core_group_1.num_cores();
    // auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);

    // SetRuntimeArgs(
    //     program,
    //     read_kernel_id,
    //     worker_grid,
    //     {
    //         (uint32_t)input_buffer->address(),
    //         (uint32_t)output_buffer->address(),
    //         (uint32_t),
    //         (uint32_t)num_units_per_core,
    //         (uint32_t)0,
    //     });
    // SetRuntimeArgs(
    //     program,
    //     write_kernel_id,
    //     worker_grid,
    //     {
    //         (uint32_t)input_buffer->address(),
    //         (uint32_t)output_buffer->address(),
    //         (uint32_t)output_unit_size,
    //         (uint32_t)num_units_per_core,
    //         (uint32_t)0,
    //     });

    return {std::move(program), {read_kernel_id, write_kernel_id, worker_grid}};
}

void DeinterleaveOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& program = cached_program.program;
    const auto& read_kernel_id = cached_program.shared_variables.read_kernel_id;
    const auto& write_kernel_id = cached_program.shared_variables.write_kernel_id;

    auto input_buffer_address = tensor_args.input.buffer()->address();
    auto output_buffer_address = output.buffer()->address();

    TT_FATAL(false, "to resolve overriding runtime args");
    // std::vector<std::vector<uint32_t>>& reader_args = GetRuntimeArgs(program, read_kernel_id);
    // reader_args[0] = input_buffer_address;
    // std::vector<std::vector<uint32_t>>& writer_args = GetRuntimeArgs(program, write_kernel_id);
    // writer_args[0] = output_buffer_address;
}
}  // namespace ttnn::operations::experimental::deinterleave
