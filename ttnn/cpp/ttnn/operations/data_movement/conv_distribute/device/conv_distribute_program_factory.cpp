// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hostdevcommon/kernel_structs.h"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/buffer_constants.hpp"
#include "tt-metalium/circular_buffer_types.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/logger.hpp"
#include "tt-metalium/small_vector.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include <algorithm>
#include <cstdint>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tuple>
#include <vector>
#include "conv_distribute_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

// New nomenclature:
// DataChunk: contiguous range of sticks from a single input core (CoreCoord, start_idx, end_idx)
// DataBlock: a block of output data, which may be composed of multiple DataChunks (vector<DataChunk>)
// CoreMapping: mapping from an output core to its assigned DataBlocks (pair<CoreCoord, vector<DataBlock>>)
using DataChunk =
    std::tuple<CoreCoord, uint32_t, uint32_t>;  // (core, start_stick, end_stick) contiguous range from a single core
using DataBlock = std::vector<DataChunk>;       // vector of DataChunks, possibly spanning multiple cores
using CoreMapping = std::pair<CoreCoord, std::vector<DataBlock>>;  // output core mapped to its assigned DataBlocks

// Moved from split_knit_utils.cpp
std::vector<CoreMapping> calculate_core_mapping(
    const ttnn::CoreRangeSet& cores,
    int block_size,
    uint32_t input_sticks_per_core,
    uint32_t output_blocks_per_core,
    uint32_t num_cores_with_extra_block,
    uint32_t total_input_sticks) {
    std::vector<CoreMapping> core_mappings;
    auto bounding_box = cores.bounding_box();
    auto size = bounding_box.size();
    auto grid_size = bounding_box.grid_size();

    // Compute how many sticks each input core actually has
    uint32_t num_input_cores = grid_size.x * grid_size.y;
    std::vector<uint32_t> sticks_per_core(num_input_cores, input_sticks_per_core);
    uint32_t sticks_left = total_input_sticks;
    for (uint32_t i = 0; i < num_input_cores; ++i) {
        if (sticks_left >= input_sticks_per_core) {
            sticks_left -= input_sticks_per_core;
        } else {
            sticks_per_core[i] = sticks_left;
            sticks_left = 0;
        }
    }

    // Prepare a flat list of (input_core, stick_idx) for all real sticks
    std::vector<std::pair<CoreCoord, uint32_t>> flat_sticks;
    for (uint32_t core_idx = 0; core_idx < num_input_cores; ++core_idx) {
        CoreCoord input_core = {core_idx % grid_size.x, core_idx / grid_size.x};
        for (uint32_t stick = 0; stick < sticks_per_core[core_idx]; ++stick) {
            flat_sticks.emplace_back(input_core, stick);
        }
    }

    // For each output DataBlock, take block_size sticks from flat_sticks, possibly spanning input cores
    size_t block_ptr = 0;
    uint32_t total_blocks = size * output_blocks_per_core + num_cores_with_extra_block;
    std::vector<DataBlock> blocks;
    for (uint32_t b = 0; b < total_blocks; ++b) {
        if (block_ptr >= flat_sticks.size()) {
            blocks.push_back({});  // skipped block
            continue;
        }
        DataBlock block;
        uint32_t sticks_in_block = 0;
        while (sticks_in_block < block_size && block_ptr < flat_sticks.size()) {
            CoreCoord input_core = flat_sticks[block_ptr].first;
            uint32_t stick = flat_sticks[block_ptr].second;
            // Group consecutive sticks from the same input core
            if (block.empty() || std::get<0>(block.back()) != input_core || std::get<2>(block.back()) != stick) {
                block.emplace_back(input_core, stick, stick + 1);
            } else {
                std::get<2>(block.back()) += 1;
            }
            ++block_ptr;
            ++sticks_in_block;
        }
        if (sticks_in_block == block_size) {
            tt::log_info("Block {}:", b);
            for (const auto& chunk : block) {
                auto input_core = std::get<0>(chunk);
                auto start_index = std::get<1>(chunk);
                auto end_index = std::get<2>(chunk);
                tt::log_info(
                    "  Chunk: input_core=({}, {}), start_index={}, end_index={}",
                    input_core.x,
                    input_core.y,
                    start_index,
                    end_index);
            }
            blocks.push_back(block);
        } else {
            tt::log_info("This shouldn't happen.");
            blocks.push_back({});  // not enough sticks left for a full block
        }
    }

    // Assign DataBlocks to output cores
    size_t block_idx = 0;
    for (uint32_t i = 0; i < size; ++i) {
        CoreCoord output_core = {i % grid_size.x, i / grid_size.x};
        std::vector<DataBlock> core_blocks;
        uint32_t blocks_this_core =
            output_blocks_per_core + ((num_cores_with_extra_block > 0 && i < num_cores_with_extra_block) ? 1 : 0);
        for (uint32_t b = 0; b < blocks_this_core; ++b) {
            if (block_idx < blocks.size() && !blocks[block_idx].empty()) {
                core_blocks.push_back(blocks[block_idx]);
            } else {
                core_blocks.push_back({});  // skipped
            }
            ++block_idx;
        }
        core_mappings.emplace_back(output_core, core_blocks);
    }
    return core_mappings;
}

operation::ProgramWithCallbacks conv_distribute_multi_core(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const MemoryConfig& distributed_mem_config,
    int block_size,
    int num_blocks_per_core,
    int num_cores_with_extra_block) {
    tt::tt_metal::Program program{};
    auto device = input_tensor.device();
    auto core_grid = distributed_mem_config.shard_spec.value().grid;

    // process input and output tensor formats
    auto input_shard_spec = input_tensor.shard_spec().value();
    uint32_t input_sticks_per_core = input_shard_spec.shape[0];
    uint32_t input_stick_size = input_shard_spec.shape[1];

    auto output_shard_spec = output_tensor.shard_spec().value();
    uint32_t output_sticks_per_core_max = output_shard_spec.shape[0];
    uint32_t output_stick_size = output_shard_spec.shape[1];

    // Use the true total number of input sticks from input_tensor.shape()[2]
    uint32_t total_input_sticks = input_tensor.get_logical_shape()[2];
    auto all_core_mappings = calculate_core_mapping(
        core_grid,
        block_size,
        input_sticks_per_core,
        num_blocks_per_core,
        num_cores_with_extra_block,
        total_input_sticks);

    // set up circular buffers
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());

    uint32_t out_cb_index = tt::CBIndex::c_0;

    tt::tt_metal::CircularBufferConfig out_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_cores_with_extra_block
                ? (num_blocks_per_core + 1) * block_size * output_stick_size * output_tensor.element_size()
                : num_blocks_per_core * block_size * output_stick_size * output_tensor.element_size(),
            {{out_cb_index, output_cb_data_format}})
            .set_page_size(out_cb_index, output_stick_size * output_tensor.element_size())
            .set_globally_allocated_address(*output_tensor.buffer());

    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, core_grid, out_cb_config);

    // create kernels
    const std::string kernel_name =
        "ttnn/cpp/ttnn/operations/data_movement/conv_distribute/device/kernels/dataflow/conv_distribute_reader.cpp";
    tt::tt_metal::KernelHandle kernel_id_0 = tt::tt_metal::CreateKernel(
        program, kernel_name, core_grid, tt::tt_metal::ReaderDataMovementConfig({out_cb_index}));
    tt::tt_metal::KernelHandle kernel_id_1 = tt::tt_metal::CreateKernel(
        program, kernel_name, core_grid, tt::tt_metal::WriterDataMovementConfig({out_cb_index}));

    uint32_t src_address = input_tensor.buffer()->address();
    // Process CoreMapping to assign runtime arguments to cores
    for (CoreMapping m : all_core_mappings) {
        auto core = m.first;
        auto data_blocks = m.second;

        // log_info(tt::LogOp, "Core {} has {} blocks", core, data_blocks.size());

        // number of chunks to read for this core
        uint32_t num_reads = 0;
        for (const auto& block : data_blocks) {
            num_reads += block.size();
            // log_info(tt::LogOp, "Block has {} chunks", block.size());
        }

        std::vector<uint32_t> runtime_args_kernel_0;
        std::vector<uint32_t> runtime_args_kernel_1;

        // Runtime arg format:
        // src_address
        // num_reads
        // For each chunk: bank_id, read_offset, write_offset, read_size
        runtime_args_kernel_0.push_back(src_address);
        runtime_args_kernel_0.push_back(num_reads);

        runtime_args_kernel_1.push_back(src_address);
        runtime_args_kernel_1.push_back(num_reads);

        uint32_t write_offset = 0;  // Shared cumulative write offset for both kernels

        for (const auto& block : data_blocks) {
            if (block.empty()) {
                // This is a skipped/empty DataBlock, increment write_offset by block_size * input_stick_size *
                // input_tensor.element_size();
                write_offset += block_size * output_stick_size * output_tensor.element_size();
                // log_info(tt::LogOp, "Skipped block");
                continue;
            }
            uint32_t block_write_offset = 0;
            for (const auto& chunk : block) {
                auto input_core = std::get<0>(chunk);
                auto start_index = std::get<1>(chunk);
                auto end_index = std::get<2>(chunk);
                uint32_t element_size = input_tensor.element_size();
                uint32_t read_size = (end_index - start_index) * output_stick_size * element_size;
                auto bank_id = device->allocator()->get_bank_ids_from_logical_core(BufferType::L1, input_core)[0];
                uint32_t read_offset = start_index * input_stick_size * element_size;
                uint32_t chunk_half_size = read_size / 2;

                if (input_core == CoreCoord(6, 3) && start_index == 320) {
                    log_info(tt::LogOp, "oops");
                    log_info(
                        tt::LogOp,
                        "Input core {}: read_offset {} write_offset {} read_size {} chunk_half_size {}",
                        input_core,
                        read_offset,
                        write_offset + block_write_offset,
                        read_size,
                        chunk_half_size);
                }

                // Chunkwise kernel arguments (for each contiguous region in the DataBlock)
                runtime_args_kernel_0.push_back(bank_id);
                runtime_args_kernel_0.push_back(read_offset);
                runtime_args_kernel_0.push_back(write_offset + block_write_offset);
                runtime_args_kernel_0.push_back(chunk_half_size);

                runtime_args_kernel_1.push_back(bank_id);
                runtime_args_kernel_1.push_back(read_offset + chunk_half_size);
                runtime_args_kernel_1.push_back(write_offset + block_write_offset + chunk_half_size);
                runtime_args_kernel_1.push_back(read_size - chunk_half_size);

                block_write_offset += read_size;
            }
            write_offset += block_size * output_stick_size * input_tensor.element_size();
        }

        SetRuntimeArgs(program, kernel_id_0, core, runtime_args_kernel_0);
        SetRuntimeArgs(program, kernel_id_1, core, runtime_args_kernel_1);
    }

    auto override_runtime_arguments_callback = [kernel_id_0, kernel_id_1, cb_output, core_grid](
                                                   const void* operation,
                                                   tt::tt_metal::Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& input = input_tensors.at(0);
        const auto& output = output_tensors.at(0);

        uint32_t input_address = input.buffer()->address();

        auto& runtime_args_0_by_core = GetRuntimeArgs(program, kernel_id_0);
        auto& runtime_args_1_by_core = GetRuntimeArgs(program, kernel_id_1);

        for (auto core : core_grid.bounding_box()) {
            auto& runtime_args_0 = runtime_args_0_by_core[core.x][core.y];
            auto& runtime_args_1 = runtime_args_1_by_core[core.x][core.y];
            runtime_args_0[0] = input_address;
            runtime_args_1[0] = input_address;
        }

        UpdateDynamicCircularBufferAddress(program, cb_output, *output.buffer());
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = nullptr};
}

}  // namespace ttnn::operations::data_movement::detail
