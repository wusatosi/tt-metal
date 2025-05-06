// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_distribute_op.hpp"
#include <optional>
#include <vector>
#include "conv_distribute_program_factory.hpp"

#include "tt-metalium/assert.hpp"
#include "tt-metalium/buffer_constants.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/logger.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void ConvDistributeDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    // These are the same requirements as conv_knit, we are keeping them for now as conv_knit is the only user
    // Potentially need to check that passed corerangesets are contigious and that there is a corresponding shard size
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Conv distribute operand needs to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Conv distribute operand needs to be allocated in buffers on device!");

    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Conv distribute operand needs to be height sharded");
    TT_FATAL(input_tensor.memory_config().buffer_type == BufferType::L1, "Conv distribute operand needs to be in L1");
    TT_FATAL(input_tensor.get_layout() == Layout::ROW_MAJOR, "Conv distribute operand needs to be row major");
    TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16, "Conv distribute operand needs to be BFLOAT16");

    // Tensor shape needs to be in format
    // [1, 1, N * H * W, C]
    TT_FATAL(
        input_tensor.get_logical_shape().to_array_4D()[0] == 1 &&
            input_tensor.get_logical_shape().to_array_4D()[1] == 1,
        "Conv distribute operand shape needs to be in format: [1, 1, N * H * W, C]");
}

std::vector<ttnn::TensorSpec> ConvDistributeDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    // TODO: compute output specs based on passed CoreRangeSet and shard sizes
    auto input_tensor = input_tensors.at(0);
    auto input_shape = input_tensor.get_logical_shape();

    // Calculate core data distribution
    uint32_t nhw = input_shape[2];
    uint32_t c = input_shape[3];

    uint32_t num_cores = this->cores.num_cores();

    // TODO: less confusing nomenclature?
    uint32_t evenly_divisible_blocks = nhw / num_cores / this->divisor;
    uint32_t evenly_divisible_extra_rows = nhw / num_cores % this->divisor;
    uint32_t remainder_blocks = nhw % num_cores / this->divisor;
    uint32_t remainder_extra_rows = nhw % num_cores % this->divisor;

    uint32_t total_extra_blocks =
        remainder_blocks + (evenly_divisible_extra_rows * num_cores + remainder_extra_rows) / this->divisor;

    // could not prove this is zero so we calculate it to cover a potential edge case
    uint32_t evenly_divisible_extra_blocks = total_extra_blocks / num_cores;
    uint32_t remainder_extra_blocks = total_extra_blocks % num_cores;

    this->num_blocks_per_core = evenly_divisible_blocks + evenly_divisible_extra_blocks;
    this->num_cores_with_extra_block = remainder_extra_blocks;

    log_info(
        tt::LogOp,
        "Num cores: {} num_blocks_per_core: {} num_cores_with_extra_block: {}",
        num_cores,
        this->num_blocks_per_core,
        this->num_cores_with_extra_block);

    // output tensor shards are equal size to the largest number of rows on a core
    auto output_logical_shape =
        ttnn::Shape({1, 1, divisor * (num_cores * this->num_blocks_per_core + this->num_cores_with_extra_block), c});

    std::optional<std::array<uint32_t, 2>> output_shard_shape = std::nullopt;
    if (this->num_blocks_per_core == 0) {
        output_shard_shape = {this->num_cores_with_extra_block, c};
    } else {
        output_shard_shape = {this->num_blocks_per_core + 1, c};
    }

    auto output_mem_config = create_sharded_memory_config(
        output_logical_shape,
        input_tensor.memory_config().shard_spec.value().grid,
        ShardStrategy::HEIGHT,
        input_tensor.memory_config().shard_spec.value().orientation,
        output_shard_shape,
        input_tensor.get_layout());

    log_info(tt::LogOp, "in cpp: output shard spec shape: {}", output_mem_config.shard_spec.value());
    log_info(tt::LogOp, "output_mem_config: {}", output_mem_config);

    return {TensorSpec(
        output_logical_shape,
        TensorLayout(input_tensor.dtype(), PageConfig(input_tensor.get_layout()), output_mem_config))};
}

operation::ProgramWithCallbacks ConvDistributeDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return detail::conv_distribute_multi_core(
        input_tensor,
        output_tensor,
        this->cores,
        this->divisor,
        this->num_blocks_per_core,
        this->num_cores_with_extra_block);
}

}  // namespace ttnn::operations::data_movement
