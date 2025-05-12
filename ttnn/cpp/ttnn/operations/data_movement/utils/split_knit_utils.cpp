// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "split_knit_utils.hpp"

std::vector<CoreMapping> calculate_core_mapping(
    const ttnn::CoreRangeSet& cores,
    int block_size,
    uint32_t input_sticks_per_core,
    uint32_t output_blocks_per_core,
    uint32_t num_cores_with_extra_block) {
    std::vector<CoreMapping> core_mappings;

    // core grid we are working on
    auto bounding_box = cores.bounding_box();
    auto size = bounding_box.size();
    auto grid_size = bounding_box.grid_size();

    // variables to track inputs
    uint32_t inputs_remaining = input_sticks_per_core * block_size;
    uint32_t input_core_index = 0;
    uint32_t input_start_index = 0;

    // variables to track outputs
    uint32_t outputs_remaining;
    CoreCoord output_core;

    // iterate over output cores
    for (auto i = 0; i < size; i++) {
        outputs_remaining = (num_cores_with_extra_block > 0) ? (output_blocks_per_core + 1) * block_size
                                                             : output_blocks_per_core * block_size;
        output_core = {i % grid_size.x, i / grid_size.x};

        // calculate chunks for this output core
        std::vector<DataChunk> core_data_chunks;
        while (outputs_remaining > 0) {
            uint32_t chunk_size = std::min(outputs_remaining, inputs_remaining);

            CoreCoord input_core = {input_core_index % grid_size.x, input_core_index / grid_size.x};
            uint32_t input_end_index = input_start_index + chunk_size;
            core_data_chunks.emplace_back(input_core, input_start_index, input_end_index);

            outputs_remaining -= chunk_size;
            inputs_remaining -= chunk_size;

            // reset input tracking variables if all inputs for this core are processed
            if (inputs_remaining == 0) {
                input_core_index++;
                input_start_index = 0;
                inputs_remaining = input_sticks_per_core * block_size;
            } else {
                input_start_index = input_end_index;
            }
        }

        if (num_cores_with_extra_block > 0) {
            num_cores_with_extra_block--;
        }

        CoreMapping core_mapping = {output_core, core_data_chunks};
        core_mappings.push_back(core_mapping);
    }

    return core_mappings;
}

std::vector<uint32_t> calculate_shard_sizes(
    const ttnn::CoreRangeSet& cores,
    int block_size,
    uint32_t input_sticks_per_core,
    uint32_t output_blocks_per_core,
    uint32_t num_cores_with_extra_block) {
    // Get core mappings using calculate_core_mapping
    auto core_mappings = calculate_core_mapping(
        cores, block_size, input_sticks_per_core, output_blocks_per_core, num_cores_with_extra_block);

    // Initialize shard sizes for each core
    std::vector<uint32_t> shard_sizes(cores.bounding_box().size(), 0);

    // Iterate over core mappings and calculate shard sizes
    for (const auto& core_mapping : core_mappings) {
        const auto& core = core_mapping.first;
        const auto& data_chunks = core_mapping.second;

        uint32_t core_index = core.y * cores.bounding_box().grid_size().x + core.x;

        for (const auto& chunk : data_chunks) {
            uint32_t start_index = std::get<1>(chunk);
            uint32_t end_index = std::get<2>(chunk);
            shard_sizes[core_index] += (end_index - start_index);
        }
    }

    return shard_sizes;
}
