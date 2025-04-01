// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn {
namespace operations::data_movement {

struct MyNewOpOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const MemoryConfig& sharded_memory_config,
        const std::optional<DataType>& data_type_arg,
        const std::optional<bool>& keep_l1_aligned = std::nullopt);
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const std::variant<CoreCoord, CoreRangeSet>& grid,
        const std::array<uint32_t, 2> shard_shape,
        const TensorMemoryLayout shard_scheme,
        const tt::tt_metal::ShardOrientation shard_orientation,
        const std::optional<DataType>& data_type_arg,
        const std::optional<bool>& keep_l1_aligned = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto my_new_op = ttnn::
    register_operation_with_auto_launch_op<"ttnn::my_new_op", ttnn::operations::data_movement::MyNewOpOperation>();
}  // namespace ttnn
