// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_distribute.hpp"
#include "device/conv_distribute_op.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor ConvDistributeOperation::invoke(
    QueueId queue_id, const ttnn::Tensor& input_tensor, const ttnn::CoreRangeSet& cores, int divisor) {
    log_info(tt::LogOp, "in ConvDistributeInvoke queue_id: {}, cores: {}, divisor: {}", *queue_id, cores, divisor);
    return operation::run(ConvDistributeDeviceOperation{.cores = cores, .divisor = divisor}, {input_tensor}).at(0);
}
}  // namespace ttnn::operations::data_movement
