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
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/math.hpp"
#include <cstdint>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_align.hpp>
#include "conv_distribute_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks conv_distribute_multi_core(
    const Tensor& input_tensor, const Tensor& output_tensor, const ttnn::CoreRangeSet& cores, int divisor) {
    tt::tt_metal::Program program{};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());

    uint32_t nhw = input_tensor.logical_shape()[2];
    uint32_t c = input_tensor.logical_shape()[3];

    uint32_t num_cores = cores.num_cores();

    // TODO: less confusing nomenclature?
    uint32_t evenly_divisible_blocks = nhw / num_cores / divisor;
    uint32_t evenly_divisible_extra_rows = nhw / num_cores % divisor;
    uint32_t remainder_blocks = nhw % num_cores / divisor;
    uint32_t remainder_extra_rows = nhw % num_cores % divisor;

    uint32_t total_extra_blocks =
        remainder_blocks + (evenly_divisible_extra_rows * num_cores + remainder_extra_rows) / divisor;

    // could not prove this is zero so we calculate it to cover a potential edge case
    uint32_t evenly_divisible_extra_blocks = total_extra_blocks / num_cores;
    uint32_t remainder_extra_blocks = total_extra_blocks % num_cores;

    uint32_t num_blocks_per_core = evenly_divisible_blocks + evenly_divisible_extra_blocks;
    uint32_t num_cores_with_extra_block = remainder_extra_blocks;

    log_info(
        tt::LogOp,
        "Num cores: {} num_blocks_per_core: {} num_cores_with_extra_block: {}",
        num_cores,
        num_blocks_per_core,
        num_cores_with_extra_block);

    // TODO: process input and output shard specs

    // TODO: set up circular buffers

    // TODO: create kernels and distribute work to cores

    return {.program = std::move(program), .override_runtime_arguments_callback = nullptr};
}

}  // namespace ttnn::operations::data_movement::detail
