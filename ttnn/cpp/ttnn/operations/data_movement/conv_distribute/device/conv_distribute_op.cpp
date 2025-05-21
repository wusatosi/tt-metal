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
    auto input_tensor = input_tensors.at(0);
    auto input_shape = input_tensor.get_logical_shape();
    uint32_t num_cores = this->distributed_mem_config.shard_spec.value().num_cores();
    uint32_t num_sticks_per_core = this->distributed_mem_config.shard_spec.value().shape[0];
    uint32_t num_channels = input_shape[3];

    auto output_logical_shape = ttnn::Shape({1, 1, num_cores * num_sticks_per_core, num_channels});

    return {TensorSpec(
        output_logical_shape,
        TensorLayout(input_tensor.dtype(), PageConfig(input_tensor.get_layout()), this->distributed_mem_config))};
}

operation::ProgramWithCallbacks ConvDistributeDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return detail::conv_distribute_multi_core(
        input_tensor,
        output_tensor,
        this->distributed_mem_config,
        this->block_size,
        this->num_blocks_per_core,
        this->num_cores_with_extra_block);
}

}  // namespace ttnn::operations::data_movement
