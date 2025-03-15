// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/conv_crop/device/conv_crop_op.hpp"
#include "ttnn/run_operation.hpp"
#include "conv_crop.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor ConvCropOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operation::run(
               ConvCropDeviceOperation{.output_mem_config = memory_config},
               {input_tensor},
               {},
               {optional_output_tensor})
        .at(0);
}

}  // namespace ttnn::operations::data_movement
