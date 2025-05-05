// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ConvDistributeOperation {
    static Tensor invoke(QueueId queue_id, const ttnn::Tensor& input_tensor, const CoreRangeSet& cores, int divisor);
};

}  // namespace operations::data_movement

constexpr auto conv_distribute = ttnn::register_operation_with_auto_launch_op<
    "ttnn::conv_distribute",
    ttnn::operations::data_movement::ConvDistributeOperation>();
}  // namespace ttnn
