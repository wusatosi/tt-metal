// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::normalization {
struct BatchNorm {
    static Tensor invoke(
        const Tensor& input,
        const Tensor& batch_mean,
        std::optional<Tensor> output,
        const std::optional<MemoryConfig>& memory_config);
};
}  // namespace operations::normalization

constexpr auto batch_norm =
    ttnn::register_operation_with_auto_launch_op<"ttnn::batch_norm", ttnn::operations::normalization::BatchNorm>();
}  // namespace ttnn
