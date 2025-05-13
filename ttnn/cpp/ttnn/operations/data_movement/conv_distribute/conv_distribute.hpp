// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt-metalium/buffer.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ConvDistributeOperation {
    static Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const tt::tt_metal::MemoryConfig& distributed_mem_config,
        int block_size,
        int num_blocks_per_core,
        int num_cores_with_extra_block);
};

}  // namespace operations::data_movement

constexpr auto conv_distribute = ttnn::register_operation_with_auto_launch_op<
    "ttnn::conv_distribute",
    ttnn::operations::data_movement::ConvDistributeOperation>();
}  // namespace ttnn
