// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"

namespace ttnn::operations::examples {
ExampleDeviceOperation::MultiCore::cached_program_t ExampleDeviceOperation::MultiCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    uint32_t single_tile_size_output = tt::tt_metal::detail::TileSize(cb_data_format_output);

    uint32_t num_tiles = input_tensor.volume() / tt::constants::TILE_HW;

    tt::tt_metal::Device* device = input_tensor.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = tt::CBIndex::c_2;
    uint32_t num_output_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size_output, {{output_cb_index, cb_data_format_output}})
            .set_page_size(output_cb_index, single_tile_size_output);
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram};
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        num_tiles_per_core_group_1,  // per_core_block_cnt
        1                            // per_core_block_size
    };

    bool math_approx_mode = false;
    auto eltwise_unary_kernel_group_1_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp",
        core_group_1,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args_group_1});

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            num_tiles_per_core_group_2,  // per_core_block_cnt
            1                            // per_core_block_size
        };

        auto eltwise_unary_kernel_group_2_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp",
            core_group_2,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args_group_2});
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt::tt_metal::SetRuntimeArgs(
            program, unary_reader_kernel_id, core, {src_buffer->address(), num_tiles_per_core, num_tiles_written});

        tt::tt_metal::SetRuntimeArgs(
            program, unary_writer_kernel_id, core, {dst_buffer->address(), num_tiles_per_core, num_tiles_written});
        num_tiles_written += num_tiles_per_core;
    }

    std::vector<uint32_t> reader_common_args = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    };

    SetCommonRuntimeArgs(program, unary_reader_kernel_id, reader_common_args);

    return {
        std::move(program),
        {.unary_reader_kernel_id = unary_reader_kernel_id,
         .unary_writer_kernel_id = unary_writer_kernel_id,
         .cb_src0 = cb_src0,
         .num_cores_y = num_cores_y,
         .num_cores_x = num_cores_x}};
}

void ExampleDeviceOperation::MultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& cb_src0 = cached_program.shared_variables.cb_src0;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;
    auto& num_cores_x = cached_program.shared_variables.num_cores_x;

    const auto& input = tensor_args.input_tensor;
    auto& output = tensor_return_value;

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();

    uint32_t num_tiles = input.volume() / tt::constants::TILE_HW;

    auto* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    auto& cached_reader_args = GetRuntimeArgs(program, unary_reader_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, unary_writer_kernel_id);

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        {
            auto& runtime_args = cached_reader_args.at(core.x).at(core.y);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = num_tiles_per_core;
            runtime_args[2] = num_tiles_written;
        }

        {
            auto& runtime_args = cached_writer_args.at(core.x).at(core.y);
            runtime_args[0] = dst_buffer->address();
            runtime_args[1] = num_tiles_per_core;
            runtime_args[2] = num_tiles_written;
        }
        num_tiles_written += num_tiles_per_core;
    }

    auto& reader_common_args = GetCommonRuntimeArgs(program, unary_reader_kernel_id);

    uint32_t num_tiles_written = 1;
    reader_common_args[0] = num_tiles_written;
    reader_common_args[1] = num_tiles_written;
    reader_common_args[2] = num_tiles_written;
    reader_common_args[3] = num_tiles_written;
    reader_common_args[4] = num_tiles_written;
    reader_common_args[5] = num_tiles_written;
    reader_common_args[6] = num_tiles_written;
    reader_common_args[7] = num_tiles_written;
    reader_common_args[8] = num_tiles_written;
    reader_common_args[9] = num_tiles_written;
    reader_common_args[10] = num_tiles_written;

    reader_common_args[11] = num_tiles_written;
    reader_common_args[12] = num_tiles_written;
    reader_common_args[13] = num_tiles_written;
    reader_common_args[14] = num_tiles_written;
    reader_common_args[15] = num_tiles_written;
    reader_common_args[16] = num_tiles_written;
    reader_common_args[17] = num_tiles_written;
    reader_common_args[18] = num_tiles_written;
    reader_common_args[19] = num_tiles_written;

    reader_common_args[20] = num_tiles_written;
    reader_common_args[21] = num_tiles_written;
    reader_common_args[22] = num_tiles_written;
    reader_common_args[23] = num_tiles_written;
    reader_common_args[24] = num_tiles_written;
    reader_common_args[25] = num_tiles_written;
    reader_common_args[26] = num_tiles_written;
    reader_common_args[27] = num_tiles_written;
    reader_common_args[28] = num_tiles_written;
    reader_common_args[29] = num_tiles_written;
}

}  // namespace ttnn::operations::examples
