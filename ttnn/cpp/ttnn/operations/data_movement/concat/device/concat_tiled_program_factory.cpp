// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/ttnn/operations/data_movement/concat/device/concat_program_factory.hpp"

#include <algorithm>
#include <numeric>

#include "cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/tt_align.hpp>

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

std::map<ttnn::DataType, uint32_t> dt_to_size = {
    {ttnn::DataType::BFLOAT16, 2},
    {ttnn::DataType::FLOAT32, 4},
    {ttnn::DataType::UINT32, 4},
    {ttnn::DataType::UINT8, 1},
};

tt_metal::operation::ProgramWithCallbacks tiled_concat_multi_core(
    const std::vector<Tensor>& input_tensors, const uint32_t dim, const Tensor& output) {
    tt_metal::Program program = tt_metal::CreateProgram();

    tt_metal::IDevice* device = output.device();

    const tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    const bool rm_layout =
        output.get_layout() ==
        Layout::ROW_MAJOR;  // Need to make sure this is changed such that output layout is ALWAYS Tiled
    if (rm_layout) {
        std::cout << "Warning: Output layout is ROW_MAJOR, but expected TILED" << std::endl;
    } else {
        std::cout << "Output layout is TILED" << std::endl;
    }

    uint32_t problem_size;
    uint32_t single_page_size;
    problem_size = output.volume();  // num elements to write, split these between cores
    uint32_t height = output.get_padded_shape()[-2];
    uint32_t width = output.get_padded_shape()[-1];
    uint32_t num_dims = output.get_padded_shape().rank();
    uint32_t input_element_size_bytes = dt_to_size[output.get_dtype()];
    uint32_t single_face_row_size = FACE_HEIGHT * input_element_size_bytes;
    uint32_t cb_page_size = single_face_row_size + sizeof(uint16_t);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Split total work among cores
    auto [num_cores, all_cores, core_group_1, core_group_2, num_elems_per_core_group_1, num_elems_per_core_group_2] =
        tt_metal::split_work_to_cores(compute_with_storage_grid_size, problem_size);

    uint32_t num_input_tensors = input_tensors.size();

    tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // Set up circular buffer
    uint32_t src0_cb_index = 0;
    uint32_t cb_multiplier = 2;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(cb_multiplier * cb_page_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, cb_page_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    // Calculate tiling parameters
    uint32_t padded_height = tt::div_up(height, TILE_HEIGHT) * TILE_HEIGHT;
    uint32_t padded_width = tt::div_up(width, TILE_WIDTH) * TILE_WIDTH;
    uint32_t tiles_per_2d_tensor = (padded_height / TILE_HEIGHT) * (padded_width / TILE_WIDTH);
    uint32_t tiles_per_tile_row = padded_width / TILE_WIDTH;

    // Reader compile-time args
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        src0_cb_index,              // cb_id_0
        dst_is_dram,                // tensor_in_dram
        input_element_size_bytes,   // element_size_bytes
        height,                     // logical_height
        width,                      // logical_width
        padded_height,              // padded_height
        padded_width,               // padded_width
        tiles_per_2d_tensor,        // tiles_per_2d_tensor
        tiles_per_tile_row,         // tiles_per_tile_row
        num_input_tensors,          // num_input_tensors
        TILE_HEIGHT,                // tile_size
        FACE_HEIGHT,                // face_size
        output.buffer()->address()  // dst_addr
    };

    std::map<string, string> concat_defines;

    if (dim == num_dims - 1) {
        concat_defines["WIDTH_CONCAT"] = "1";
    }

    std::vector<uint32_t> writer_compile_time_args = {// interleaved accessor args
                                                      (std::uint32_t)src0_cb_index,
                                                      (std::uint32_t)dst_is_dram};

    // Tilized reader
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
        "reader_concat_tiled.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, concat_defines));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
        "writer_s2i_width.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    const auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);
    uint32_t g1_num_cores = core_group_1.num_cores();
    uint32_t elems_processed = 0;
    uint32_t curr_tensor = 0;

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        uint32_t num_elems_per_core = (i < g1_num_cores) ? num_elems_per_core_group_1 : num_elems_per_core_group_2;

        std::vector<uint32_t> reader_runtime_args;

        reader_runtime_args.push_back(cb_page_size);
        reader_runtime_args.push_back(num_elems_per_core);  // How many elements this core processes

        // Calculate local tile offset in starting tensor
        // reader_runtime_args.push_back(local_tile_offset);
        reader_runtime_args.push_back(curr_tensor);  // Which tensor to start with
        // Calculate starting row and column even if tensor is N dimensional
        uint32_t starting_row = elems_processed / input_tensors[curr_tensor].get_logical_shape()[-1];
        uint32_t starting_col = elems_processed % input_tensors[curr_tensor].get_logical_shape()[-1];
        reader_runtime_args.push_back(starting_row);
        reader_runtime_args.push_back(starting_col);

        // Add tensor addresses
        for (const auto& tensor : input_tensors) {
            reader_runtime_args.push_back(tensor.buffer()->address());
            reader_runtime_args.push_back(tensor.volume());
            uint32_t input_height = tensor.get_logical_shape()[-2];
            uint32_t input_width = tensor.get_logical_shape()[-1];
            reader_runtime_args.push_back(input_height);
            reader_runtime_args.push_back(input_width);
        }

        // make reader runtime args
        tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
        elems_processed += num_elems_per_core;
        if (elems_processed >= input_tensors[curr_tensor].volume()) {
            elems_processed -= input_tensors[curr_tensor].volume();
            curr_tensor++;
        }
    }

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id, cores](
                                              const void* operation,
                                              const Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        std::vector<uint32_t> src_addrs(input_tensors.size());
        for (uint32_t i = 0; i < input_tensors.size(); ++i) {
            src_addrs[i] = input_tensors[i].buffer()->address();
        }

        auto dst_buffer = output_tensors.at(0).buffer();

        for (const auto& core : cores) {
            {
                auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
                std::copy(src_addrs.begin(), src_addrs.end(), runtime_args.data() + 3);
            }

            {
                auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement::detail
