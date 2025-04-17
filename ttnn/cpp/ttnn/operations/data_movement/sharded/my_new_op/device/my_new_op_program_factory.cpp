// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "cpp/ttnn/operations/data_movement/sharded/sharded_common.hpp"
#include "cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op.hpp"
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/hal.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks my_new_op_multi_core(
    const Tensor& input,
    const Tensor& input2,
    const Tensor& output,
    float scalar_multiplier,
    uint32_t num_slices,
    uint32_t slice_index) {
    tt::tt_metal::Program program{};
    // bool keep_l1_aligned = true;
    // uint32_t num_units, num_units_per_shard, input_unit_size, output_unit_size, num_units_per_shard_width,
    //     num_units_per_shard_height, num_units_offset, num_units_per_row, num_units_per_shard_height_last,
    //     num_units_per_shard_width_last, padded_offset_bytes;

    std::cout << "*************** my_new_op_multi_core *******************" << std::endl;
    std::cout << "scalar_multiplier = " << scalar_multiplier << std::endl;

    tt::tt_metal::IDevice* device = input.device();

    auto src0_buffer = input.buffer();
    auto src1_buffer = input2.buffer();
    auto dst_buffer = output.buffer();

    // calculate tile sizes and number of tiles
    tt::DataFormat src0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat src1_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input2.get_dtype());
    uint32_t single_tile_size_input2 = tt::tt_metal::detail::TileSize(src1_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t single_tile_size_output = tt::tt_metal::detail::TileSize(output_cb_data_format);

    uint32_t num_units = input.volume() / tt::constants::TILE_HW;

    // split work to cores
    CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    uint32_t num_tiles = input.volume() / tt::constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    // source and output buffers for reader and writer kernels
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t out_cb_index = tt::CBIndex::c_2;

    uint32_t mul_cb_index = tt::CBIndex::c_3;

    uint32_t num_input_tiles = num_tiles * 1;   //???
    uint32_t num_output_tiles = num_tiles * 1;  //???

    std::cout << "src0_cb_index: " << src0_cb_index << std::endl;
    std::cout << "src1_cb_index: " << src1_cb_index << std::endl;
    std::cout << "out_cb_index: " << out_cb_index << std::endl;
    std::cout << "num_input_tiles: " << num_input_tiles << std::endl;
    std::cout << "num_output_tiles: " << num_output_tiles << std::endl;

    uint32_t num_input_units = 1;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(num_input_units * single_tile_size, {{src1_cb_index, src1_cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_units * single_tile_size_output, {{out_cb_index, output_cb_data_format}})
            .set_page_size(out_cb_index, single_tile_size_output);
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);
    std::cout << "output_cb_out_config: " << std::endl;

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src0_is_dram,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)src1_is_dram,
        1};  // all_cores.num_cores()};
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)out_cb_index, (std::uint32_t)dst_is_dram, 1};

    std::cout << "src0_buffer: " << src0_is_dram << std::endl;
    std::cout << "src1_buffer: " << src1_is_dram << std::endl;
    std::cout << "dst_buffer: " << dst_is_dram << std::endl;

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "/localdev/bjanjic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/sharded/my_new_op/device/kernels/dataflow/"
        "reader.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "/localdev/bjanjic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/sharded/my_new_op/device/kernels/dataflow/"
        "writer.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // create compute kernel
    uint32_t casted_scalar = std::bit_cast<uint32_t>(scalar_multiplier);
    std::cout << "casted_scalar: " << casted_scalar << std::endl;
    std::vector<uint32_t> compute_compile_time_args_1 = {
        num_tiles_per_core_group_1,
        16,
        casted_scalar,  // 2.0f in uint32_t
        src0_cb_index,
        src1_cb_index,
        out_cb_index,
        mul_cb_index};
    std::vector<uint32_t> compute_compile_time_args_2 = {
        num_tiles_per_core_group_2,
        16,
        casted_scalar,  // 2.0f in uint32_t
        src0_cb_index,
        src1_cb_index,
        out_cb_index,
        mul_cb_index};

    auto compute_kernel_1 = tt::tt_metal::CreateKernel(
        program,
        "/localdev/bjanjic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/sharded/my_new_op/device/kernels/dataflow/"
        "compute.cpp",
        core_group_1,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args_1});

    auto compute_kernel_2 = (core_group_2.ranges().empty())
                                ? compute_kernel_1
                                : tt::tt_metal::CreateKernel(
                                      program,
                                      "/localdev/bjanjic/tt-metal/ttnn/cpp/ttnn/operations/data_movement/sharded/"
                                      "my_new_op/device/kernels/dataflow/compute.cpp",
                                      core_group_2,
                                      tt::tt_metal::ComputeConfig{
                                          .math_fidelity = MathFidelity::HiFi4,
                                          .math_approx_mode = false,
                                          .compile_args = compute_compile_time_args_2});

    // uint32_t starting_idx_h = calculate_starting_idx_h(input, num_slices, slice_index);
    // uint32_t curr_idx_h = 0;
    // uint32_t curr_idx_w = 0;

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
            {src0_buffer->address(), num_tiles_written, src1_buffer->address(), num_tiles_written, num_tiles_per_core});

        tt::tt_metal::SetRuntimeArgs(
            program, unary_writer_kernel_id, core, {dst_buffer->address(), num_tiles_written, num_tiles_per_core});
        num_tiles_written += num_tiles_per_core;
    }

    auto override_runtime_arguments_callback =
        [unary_reader_kernel_id, unary_writer_kernel_id, cb_output, num_cores, num_cores_y](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>&,
            const std::vector<Tensor>& output_tensors) {
            auto src0_buffer = input_tensors.at(0).buffer();
            auto src1_buffer = input_tensors.at(1).buffer();
            auto dst_buffer = output_tensors.at(0).buffer();

            auto& runtime_args_by_core = GetRuntimeArgs(program, unary_reader_kernel_id);
            for (uint32_t core = 0; core < num_cores; core++) {
                std::cout << "core: " << num_cores << std::endl;
                // tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
                //  {
                //  auto& runtime_args = runtime_args_by_core[core.x][core.y];
                //  runtime_args[0] = src0_buffer->address();
                //  runtime_args[2] = src1_buffer->address();
                //  // if (partial_op) {
                //  //     runtime_args[7] = starting_idx_h;
                //  // }
                //  }
            }
            UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::data_movement::detail
