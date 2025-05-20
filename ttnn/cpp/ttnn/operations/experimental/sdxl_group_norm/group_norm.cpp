// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/group_norm_op.hpp"
#include "ttnn/operations/experimental/sdxl_group_norm/group_norm.hpp"

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor SDXLGroupNormOperation::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const Tensor& weights,
    const Tensor& bias,
    const float eps,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return tt::tt_metal::operation::run(
               SDXLGroupNorm{eps, compute_kernel_config, sub_core_grids},
               {input_tensor, weights, bias},
               {},
               {},
               queue_id)
        .at(0);
}

}  // namespace ttnn::operations::experimental
