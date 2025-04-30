// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_distribute_op.hpp"
#include <vector>
#include "conv_distribute_program_factory.hpp"

#include "tt-metalium/assert.hpp"
#include "tt-metalium/buffer_constants.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/logger.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/functions.hpp"
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

    // placeholder output to compile
    auto input_tensor = input_tensors.at(0);
    auto output_tensor_spec = input_tensor.get_tensor_spec();
    std::vector<ttnn::TensorSpec> return_value;
    return_value.push_back(output_tensor_spec);

    return (return_value);
}

operation::ProgramWithCallbacks ConvDistributeDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return detail::conv_distribute_multi_core(input_tensor, output_tensor, this->cores, this->shard_sizes);
}

}  // namespace ttnn::operations::data_movement
