// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include "ttnn/operation.hpp"

#include <iostream>
#include <cmath>

using namespace tt::tt_metal;
using namespace std;
namespace ttnn::operations::reduction::detail {

operation::ProgramWithCallbacks topk_single_core_interleaved(
    const Tensor& input_tensor,
    const uint32_t k,
    const int8_t dim,
    const bool largest,
    const bool sorted,
    Tensor& value_tensor,
    Tensor& index_tensor) {
    using namespace tt::constants;
    tt::tt_metal::Program program{};
    CoreRange core({0, 0}, {0, 0});
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::DataFormat output_val_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(value_tensor.get_dtype());
    tt::DataFormat output_ind_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(index_tensor.get_dtype());

    // Print function introduction
    std::cout << "TopK Single Core Interleaved" << std::endl;
    std::cout << "  Input tensor shape: " << input_tensor.get_padded_shape()[0] << " "
              << input_tensor.get_padded_shape()[1] << " " << input_tensor.get_padded_shape()[2] << " "
              << input_tensor.get_padded_shape()[3] << std::endl;
    std::cout << "  k: " << k << std::endl;
    std::cout << "  dim: " << dim << std::endl;
    std::cout << "  largest: " << largest << std::endl;
    std::cout << "  sorted: " << sorted << std::endl;
    std::cout << std::flush;

    uint32_t input_tile_size = tile_size(input_cb_data_format);
    uint32_t value_tile_size = tile_size(output_val_cb_data_format);
    uint32_t index_tile_size = tile_size(output_ind_cb_data_format);

    // Print tile sizes
    std::cout << "  Input tile size: " << input_tile_size << std::endl;
    std::cout << "  Value tile size: " << value_tile_size << std::endl;
    std::cout << "  Index tile size: " << index_tile_size << std::endl;
    std::cout << std::flush;

    auto input_buffer = input_tensor.buffer();
    auto values_buffer = value_tensor.buffer();
    auto index_buffer = index_tensor.buffer();

    bool input_is_dram = input_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool values_is_dram = values_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool index_is_dram = index_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    // Print buffer types
    std::cout << "  Input is DRAM: " << input_is_dram << std::endl;
    std::cout << "  Values is DRAM: " << values_is_dram << std::endl;
    std::cout << "  Index is DRAM: " << index_is_dram << std::endl;
    std::cout << std::flush;

    uint32_t num_input_tiles = input_tensor.volume() / TILE_HW;
    uint32_t num_value_tiles = value_tensor.volume() / TILE_HW;

    // Print number of tiles
    std::cout << "  Number of input tiles: " << num_input_tiles << std::endl;
    std::cout << "  Number of value tiles: " << num_value_tiles << std::endl;
    std::cout << std::flush;

    auto input_shape = input_tensor.get_padded_shape();
    uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / TILE_HEIGHT;
    uint32_t Wt = input_shape[3] / TILE_WIDTH;

    uint32_t Ktiles = (k + 31) / 32;
    std::cout << "Ktiles: " << Ktiles << std::endl;

    // Print Ht and Wt
    std::cout << "  Ht (height in tiles): " << Ht << std::endl;
    std::cout << "  Wt (width in tiles): " << Wt << std::endl;
    std::cout << std::flush;

    // for streaming in input
    uint32_t num_cb_unit = 2;
    uint32_t cb_in_units = 2 * num_cb_unit;

    uint32_t input_cb_tile_count = cb_in_units;
    uint32_t transposed_cb_tile_count = 2 * input_cb_tile_count;
    uint32_t result_prep_cb_tile_count = 2 * Ktiles;  // intermediate output
    uint32_t output_cb_tile_count = 2 * Ktiles;       // final output

    // Print circular buffer configuration
    std::cout << "  Number of CB units: " << num_cb_unit << std::endl;
    std::cout << "  CB in units: " << cb_in_units << std::endl;
    std::cout << std::flush;

    // Two tiles are loaded in for topk_local_sort at a time, and we double buffer to avoid stalls, so allocate four
    // tiles of space
    // TODO: In theory if we have enough memory we could allocate 2*Wt tiles to reduce stalls
    uint32_t input_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig input_cb_config =
        tt::tt_metal::CircularBufferConfig(
            input_cb_tile_count * value_tile_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, input_tile_size);
    auto cb_input_tensor = tt::tt_metal::CreateCircularBuffer(program, core, input_cb_config);

    // Two tiles are loaded in for topk_local_sort at a time, and we double buffer to avoid stalls, so allocate four
    // tiles of space. This CB carries the indices that are created in the reader kernel
    uint32_t index_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig index_input_intermed0_config =
        tt::tt_metal::CircularBufferConfig(
            input_cb_tile_count * index_tile_size, {{index_cb_index, output_ind_cb_data_format}})
            .set_page_size(index_cb_index, index_tile_size);
    auto cb_index_tensor = tt::tt_metal::CreateCircularBuffer(program, core, index_input_intermed0_config);

    // Single buffered circular buffer that holds the transposed input tiles
    uint32_t transposed_val_cb_index = tt::CBIndex::c_22;
    tt::tt_metal::CircularBufferConfig transposed_val_cb_config =
        tt::tt_metal::CircularBufferConfig(
            transposed_cb_tile_count * value_tile_size, {{transposed_val_cb_index, input_cb_data_format}})
            .set_page_size(transposed_val_cb_index, input_tile_size);
    auto cb_input_transposed_tiles = tt::tt_metal::CreateCircularBuffer(program, core, transposed_val_cb_config);

    // Single buffered circular buffer that holds the transposed index tiles
    uint32_t transposed_ind_cb_index = tt::CBIndex::c_23;
    tt::tt_metal::CircularBufferConfig transposed_ind_cb_config =
        tt::tt_metal::CircularBufferConfig(
            transposed_cb_tile_count * index_tile_size, {{transposed_ind_cb_index, output_ind_cb_data_format}})
            .set_page_size(transposed_ind_cb_index, index_tile_size);
    auto cb_index_transposed_tiles = tt::tt_metal::CreateCircularBuffer(program, core, transposed_ind_cb_config);

    // Single buffered circular buffer that holds the result_prep input tiles
    uint32_t result_prep_val_cb_index = tt::CBIndex::c_24;
    tt::tt_metal::CircularBufferConfig result_prep_val_cb_config =
        tt::tt_metal::CircularBufferConfig(
            result_prep_cb_tile_count * value_tile_size, {{result_prep_val_cb_index, input_cb_data_format}})
            .set_page_size(result_prep_val_cb_index, input_tile_size);
    auto cb_input_result_prep_tiles = tt::tt_metal::CreateCircularBuffer(program, core, result_prep_val_cb_config);

    // Single buffered circular buffer that holds the result_prep index tiles
    uint32_t result_prep_ind_cb_index = tt::CBIndex::c_25;
    tt::tt_metal::CircularBufferConfig result_prep_ind_cb_config =
        tt::tt_metal::CircularBufferConfig(
            result_prep_cb_tile_count * index_tile_size, {{result_prep_ind_cb_index, output_ind_cb_data_format}})
            .set_page_size(result_prep_ind_cb_index, index_tile_size);
    auto cb_index_result_prep_tiles = tt::tt_metal::CreateCircularBuffer(program, core, result_prep_ind_cb_config);

    // Output topk values
    uint32_t output_val_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig output_val_cb_config =
        tt::tt_metal::CircularBufferConfig(
            output_cb_tile_count * value_tile_size, {{output_val_cb_index, output_val_cb_data_format}})
            .set_page_size(output_val_cb_index, value_tile_size);
    auto cb_values_tensor = tt::tt_metal::CreateCircularBuffer(program, core, output_val_cb_config);

    // Output topk indices
    uint32_t output_ind_cb_index = tt::CBIndex::c_17;
    tt::tt_metal::CircularBufferConfig output_ind_cb_config =
        tt::tt_metal::CircularBufferConfig(
            output_cb_tile_count * index_tile_size, {{output_ind_cb_index, output_ind_cb_data_format}})
            .set_page_size(output_ind_cb_index, index_tile_size);
    auto cb_output_ind_tensor = tt::tt_metal::CreateCircularBuffer(program, core, output_ind_cb_config);

    std::vector<uint32_t> reader_compile_time_args = {input_cb_index, index_cb_index, (uint32_t)input_is_dram, Ht, Wt};
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_create_index_tensor.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {
            input_buffer->address(),
        });

    std::vector<uint32_t> writer_compile_time_args = {
        output_val_cb_index, output_ind_cb_index, (std::uint32_t)values_is_dram, (std::uint32_t)index_is_dram, Ht, k};
    tt::tt_metal::KernelHandle binary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_binary_interleaved.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    SetRuntimeArgs(
        program,
        binary_writer_kernel_id,
        core,
        {
            values_buffer->address(),
            index_buffer->address(),

        });

    std::vector<uint32_t> compute_args = {
        input_cb_index,
        index_cb_index,
        transposed_val_cb_index,
        transposed_ind_cb_index,
        result_prep_val_cb_index,
        result_prep_ind_cb_index,
        output_val_cb_index,
        output_ind_cb_index,
        Ht,
        Wt,
        k,
        (std::uint32_t)std::log2(k),
        (std::uint32_t)std::log2(input_shape[3] / k),
        (std::uint32_t)largest,
        (std::uint32_t)sorted,
    };
    tt::tt_metal::KernelHandle topk_compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp",
        core,
        tt::tt_metal::ComputeConfig{.compile_args = compute_args});

    std::cout << "Input Tensor Shape: " << input_tensor.get_padded_shape() << std::endl;
    std::cout << "K: " << k << ", Dim: " << static_cast<int>(dim) << ", Largest: " << largest << ", Sorted: " << sorted
              << std::endl;
    std::cout << "Input Tile Size: " << input_tile_size << ", Value Tile Size: " << value_tile_size
              << ", Index Tile Size: " << index_tile_size << std::endl;
    std::cout << "Input is DRAM: " << input_is_dram << ", Values is DRAM: " << values_is_dram
              << ", Index is DRAM: " << index_is_dram << std::endl;
    std::cout << "Ht: " << Ht << ", Wt: " << Wt << std::endl;
    std::cout << "Num Input Tiles: " << num_input_tiles << ", Num Value Tiles: " << num_value_tiles << std::endl;
    std::cout << "Reader Compile Time Args: ";
    for (const auto& arg : reader_compile_time_args) {
        std::cout << arg << " ";
    }
    std::cout << std::endl;
    std::cout << "Writer Compile Time Args: ";
    for (const auto& arg : writer_compile_time_args) {
        std::cout << arg << " ";
    }
    std::cout << std::endl;
    std::cout << "Compute Args: ";
    for (const auto& arg : compute_args) {
        std::cout << arg << " ";
    }
    std::cout << std::endl;

    auto override_runtime_args_callback = [unary_reader_kernel_id, binary_writer_kernel_id](
                                              const void* operation,
                                              const Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto input_buffer = input_tensors.at(0).buffer();

        auto values_buffer = output_tensors.at(0).buffer();
        auto index_buffer = output_tensors.at(1).buffer();

        CoreCoord core = {0, 0};

        {
            auto& reader_runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            reader_runtime_args[0] = input_buffer->address();

            auto& writer_runtime_args = GetRuntimeArgs(program, binary_writer_kernel_id, core);
            writer_runtime_args[0] = values_buffer->address();
            writer_runtime_args[1] = index_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

/**
 * Split the work along the width such that the width is divisible by min_dim and the number of cores used is less than
 * or equal to max_cores. Each core must have a minimum of two tiles - min_dim = 64 as that's the minimum size for the
 * llk. Return the number of cores utilized for the split, the size of the split along the width, the width on the
 * remainder core if any, and the remaining elements that the gather core has to process If less than the max number of
 * cores are used, then we can try splitting on height as well. Eg) if only 2 cores are used for the split and then 1
 * for the gather, we only need 3 cores per row. Then take cores_per_row = 3 and try to split the height such that the
 * number of cores used is less than or equal to max_cores.
 */
static inline std::tuple<uint16_t, uint16_t, uint16_t, uint16_t> cores_utilized(
    uint16_t width,
    uint16_t min_dim,
    uint16_t max_dim,
    CoreCoord grid,
    uint32_t k,
    const uint32_t l1_size,
    const uint32_t value_tile_size,
    const uint32_t index_tile_size) {
    const auto max_cores = grid.y - 1;  // reserve one core for the gather - switch to grid.x as it allows for more
                                        // cores and allow spillover to next row
    for (uint16_t split_size = max_dim; split_size >= min_dim; split_size /= 2) {
        uint16_t rem = width % split_size;
        uint16_t num_cores = width / split_size + (rem > 0);
        uint32_t memory_cost_gather =
            2 * num_cores * (value_tile_size + index_tile_size);  // gathering one index and one value tile from each
                                                                  // local core, allocating two CBs for each
        uint32_t memory_cost_local =
            (split_size / tt::constants::TILE_WIDTH) *
            (value_tile_size + index_tile_size);  // we divide the width into split_size chunks and each chunk, as well
                                                  // as a matching set of indices, is processed by a core
        if (num_cores <= max_cores && (memory_cost_gather + memory_cost_local * num_cores) < (l1_size * num_cores) &&
            num_cores > 1) {
            return {
                num_cores + 1,
                split_size,
                rem,
                num_cores * std::max(static_cast<uint32_t>(k), static_cast<uint32_t>(tt::constants::TILE_WIDTH))};
        }
    }
    return {max_cores + 1, width, 0, width * k};
}
}  // namespace ttnn::operations::reduction::detail
