// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_device_operation.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::examples {
NocInlineDeviceOperation::SingleCore::cached_program_t NocInlineDeviceOperation::SingleCore::create(
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

    tt::tt_metal::IDevice* device = input_tensor.device();

    CoreCoord compute_with_storage_grid_size = {1, 1};
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    uint32_t print_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 1;
    tt::tt_metal::CircularBufferConfig cb_print_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{print_cb_index, cb_data_format}})
            .set_page_size(print_cb_index, single_tile_size);
    auto cb_print = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_print_config);

    uint32_t output_cb_index = tt::CBIndex::c_2;
    uint32_t num_output_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size_output, {{output_cb_index, cb_data_format_output}})
            .set_page_size(output_cb_index, single_tile_size_output);
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram};
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/examples/example/device/kernels/dataflow/noc_inline_dw_write_test.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::NOC_0});

    CoreCoord writer_core = {0, 0};
    CoreCoord receiver_core = {0, 1};
    CoreCoord virtual_receiver = device->worker_core_from_logical_core(receiver_core);
    tt::tt_metal::SetRuntimeArgs(
        program, unary_writer_kernel_id, writer_core, {virtual_receiver.x, virtual_receiver.y, src_buffer->address(), dst_buffer->address()});
    return {std::move(program), {.unary_writer_kernel_id = unary_writer_kernel_id}};
}

void NocInlineDeviceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    TT_THROW("Program cache is not supported in this test.");
}

}  // namespace ttnn::operations::examples
