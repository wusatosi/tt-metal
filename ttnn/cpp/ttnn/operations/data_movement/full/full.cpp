// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_rm.hpp"
#include "device/fill_rm_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/common/constants.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor FullOperation::invoke(
    const uint8_t queue_id,
    const ttnn::SimpleShape& logical_shape,
    const float fill_value,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt) {
    auto output_memory_config = memory_config.value_or(any.memory_config());
    return operation::run_without_autoformat(
               FillRM{logical_shape, fill_value, output_memory_config}, {any}, {}, {}, queue_id)
        .at(0);
}

ttnn::Tensor FullOperation::invoke(
    const uint8_t queue_id,
    const ttnn::SimpleShape& logical_shape,
    const float fill_value,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt) {
    auto output_memory_config = memory_config.value_or(any.memory_config());
    return invoke(DefaultQueueId, logical_shape, fill_value, memory_config_arg);
}

}  // namespace ttnn::operations::data_movement
