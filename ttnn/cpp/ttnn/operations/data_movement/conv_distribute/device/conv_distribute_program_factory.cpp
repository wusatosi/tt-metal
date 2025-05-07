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
#include "ttnn/types.hpp"
#include <cstdint>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tuple>
#include "conv_distribute_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

using DataChunk = std::tuple<CoreCoord, uint32_t, uint32_t>;
using CoreMapping = std::pair<CoreCoord, std::vector<DataChunk>>;

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

operation::ProgramWithCallbacks conv_distribute_multi_core(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::CoreRangeSet& cores,
    int divisor,
    uint32_t num_blocks_per_core,
    uint32_t num_cores_with_extra_block) {
    tt::tt_metal::Program program{};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());

    // TODO: process input and output tensor formats
    auto input_shard_spec = input_tensor.shard_spec().value();
    uint32_t input_sticks_per_core = input_shard_spec.shape[0];
    uint32_t input_stick_size = input_shard_spec.shape[1];

    auto all_core_mappings =
        calculate_core_mapping(cores, divisor, input_stick_per_core, num_blocks_per_core, num_cores_with_extra_block);

    // TODO: set up circular buffers

    // TODO: create kernels

    return {.program = std::move(program), .override_runtime_arguments_callback = nullptr};
}

}  // namespace ttnn::operations::data_movement::detail
