// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_distribute.hpp"
#include "device/conv_distribute_op.hpp"
#include "tt-metalium/buffer.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor ConvDistributeOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& distributed_mem_config,
    int block_size,
    int num_blocks_per_core,
    int num_cores_with_extra_block) {
    log_info(
        tt::LogOp,
        "in ConvDistributeInvoke queue_id: {}, distributed_mem_config: {}",
        *queue_id,
        distributed_mem_config);
    log_info(
        tt::LogOp,
        "in ConvDistributeInvoke block_size: {}, num_blocks_per_core: {}, num_cores_with_extra_block {}",
        block_size,
        num_blocks_per_core,
        num_cores_with_extra_block);
    return operation::run(
               ConvDistributeDeviceOperation{
                   distributed_mem_config, block_size, num_blocks_per_core, num_cores_with_extra_block},
               {input_tensor})
        .at(0);
}
}  // namespace ttnn::operations::data_movement
