// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include "my_new_op_operation.hpp"

namespace ttnn::operations::data_movement {

MyNewOpDeviceOperation::MyDeviceProgramFactory::cached_program_t MyNewOpDeviceOperation::MyDeviceProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::constants;

    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    auto& output_tensor = tensor_return_value;

    auto src_buffer_a = input_tensor_a.buffer();
    auto src_buffer_b = input_tensor_b.buffer();
    auto dst_buffer = output_tensor.buffer();

    float scalar = operation_attributes.input_scalar;
    uint32_t casted_scalar = std::bit_cast<uint32_t>(scalar);

    tt::tt_metal::Program program{};

    tt::DataFormat cb0_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_a.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb0_data_format);
    tt::DataFormat cb1_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_b.get_dtype());
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    uint32_t single_tile_size_output = tt::tt_metal::detail::TileSize(cb_data_format_output);

    uint32_t num_tiles = input_tensor_a.volume() / tt::constants::TILE_HW;

    tt::tt_metal::IDevice* device = input_tensor_a.device();

    CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t output_cb_index = tt::CBIndex::c_2;

    std::optional<ShardSpec> shard_spec = std::nullopt;
    bool src0_sharded = input_tensor_a.memory_config().is_sharded();
    bool src1_sharded = input_tensor_b.memory_config().is_sharded();
    bool out_sharded = output_tensor.memory_config().is_sharded();

    bool block_or_width_sharded = false;

    if (src0_sharded) {
        shard_spec = input_tensor_a.shard_spec().value();
        block_or_width_sharded = input_tensor_a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
    } else if (src1_sharded) {
        shard_spec = input_tensor_b.shard_spec().value();
        block_or_width_sharded = input_tensor_b.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
    } else if (out_sharded) {
        shard_spec = output_tensor.shard_spec().value();
        block_or_width_sharded = output_tensor.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
    }

    uint32_t num_tiles_per_shard = 0;
    if (shard_spec.has_value()) {
        num_tiles_per_shard = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
    }

    uint32_t num_input_tiles = src0_sharded ? num_tiles_per_shard : num_tiles_per_core_group_1;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb0_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    if (src0_sharded) {
        cb_src0_config = cb_src0_config.set_globally_allocated_address(*input_tensor_a.buffer());
    }
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb1_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    if (src1_sharded) {
        cb_src1_config = cb_src1_config.set_globally_allocated_address(*input_tensor_b.buffer());
    }
    auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    uint32_t num_output_tiles = (out_sharded || block_or_width_sharded) ? num_tiles_per_shard : 1;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size_output, {{output_cb_index, cb_data_format_output}})
            .set_page_size(output_cb_index, single_tile_size_output);
    if (out_sharded) {
        cb_output_config = cb_output_config.set_globally_allocated_address(*output_tensor.buffer());
    }
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    std::map<string, string> reader_defines;
    if (src0_sharded) {
        reader_defines["IN0_SHARDED"] = "1";
    }
    if (src1_sharded) {
        reader_defines["IN1_SHARDED"] = "1";
    }
    std::map<string, string> writer_defines;
    if (out_sharded) {
        writer_defines["OUT_SHARDED"] = "1";
    }

    const uint32_t input0_is_dram = input_tensor_a.buffer()->buffer_type() == BufferType::DRAM ? 1 : 0;
    const uint32_t input1_is_dram = input_tensor_b.buffer()->buffer_type() == BufferType::DRAM ? 1 : 0;
    const uint32_t output_is_dram = output_tensor.buffer()->buffer_type() == BufferType::DRAM ? 1 : 0;

    std::vector<uint32_t> reader_compile_time_args = {src0_cb_index, src1_cb_index, input0_is_dram, input1_is_dram};
    std::vector<uint32_t> writer_compile_time_args = {output_cb_index, output_is_dram};

    auto reader_config = tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines);
    auto writer_config = tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/my_new_op/device/kernels/dataflow/my_new_op_reader.cpp",
        all_cores,
        reader_config);

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/my_new_op/device/kernels/dataflow/my_new_op_writer.cpp",
        all_cores,
        writer_config);

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        src0_cb_index, src1_cb_index, output_cb_index, casted_scalar, num_tiles_per_core_group_1};

    bool math_approx_mode = false;
    auto eltwise_unary_kernel_group_1_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/my_new_op/device/kernels/compute/my_new_op_compute.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args_group_1});

    auto eltwise_unary_kernel_group_2_id = eltwise_unary_kernel_group_1_id;
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            src0_cb_index, src1_cb_index, output_cb_index, casted_scalar, num_tiles_per_core_group_2};

        eltwise_unary_kernel_group_2_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/sharded/my_new_op/device/kernels/compute/my_new_op_compute.cpp",
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
            program,
            unary_reader_kernel_id,
            core,
            {src_buffer_a->address(),
             num_tiles_written,
             src_buffer_b->address(),
             num_tiles_written,
             num_tiles_per_core});
        tt::tt_metal::SetRuntimeArgs(
            program, unary_writer_kernel_id, core, {dst_buffer->address(), num_tiles_written, num_tiles_per_core});
        num_tiles_written += num_tiles_per_core;
    }

    return {
        std::move(program),
        {.reader_kernel_id = unary_reader_kernel_id,
         .writer_kernel_id = unary_writer_kernel_id,
         .compute_kernel_id_1 = eltwise_unary_kernel_group_1_id,
         .compute_kernel_id_2 = eltwise_unary_kernel_group_2_id,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y}};
}

void MyNewOpDeviceOperation::MyDeviceProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    auto& output_tensor = tensor_return_value;

    auto src_buffer_a = input_tensor_a.buffer();
    auto src_buffer_b = input_tensor_b.buffer();
    auto dst_buffer = output_tensor.buffer();

    for (uint32_t i = 0; i < num_cores; i++) {
        tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer_a->address();
            runtime_args[2] = src_buffer_b->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::data_movement
