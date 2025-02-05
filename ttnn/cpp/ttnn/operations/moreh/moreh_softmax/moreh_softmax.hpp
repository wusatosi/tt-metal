// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "cpp/ttnn/operations/moreh/moreh_softmax/device/moreh_softmax_device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_softmax {

struct MorehSoftmax {
    static Tensor invoke(
        const Tensor& input_tensor,
        uint32_t dim,
        const std::optional<Tensor>& output_tensor,
        const MorehSoftmaxOp op,
        const MorehSoftmaxOpParallelizationStrategy strategy,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
struct MorehSoftmin {
    static Tensor invoke(
        const Tensor& input_tensor,
        uint32_t dim,
        const std::optional<Tensor>& output_tensor,
        const MorehSoftmaxOp op,
        const MorehSoftmaxOpParallelizationStrategy strategy,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
struct MorehLogSoftmax {
    static Tensor invoke(
        const Tensor& input_tensor,
        uint32_t dim,
        const std::optional<Tensor>& output_tensor,
        const MorehSoftmaxOp op,
        const MorehSoftmaxOpParallelizationStrategy strategy,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};

}  // namespace ttnn::operations::moreh::moreh_softmax

namespace ttnn {
constexpr auto moreh_softmax = ttnn::register_operation_with_auto_launch_op<
    "ttnn::moreh_softmax",
    ttnn::operations::moreh::moreh_softmax::MorehSoftmax>();
constexpr auto moreh_softmin = ttnn::register_operation_with_auto_launch_op<
    "ttnn::moreh_softmin",
    ttnn::operations::moreh::moreh_softmax::MorehSoftmin>();
constexpr auto moreh_logsoftmax = ttnn::register_operation_with_auto_launch_op<
    "ttnn::moreh_logsoftmax",
    ttnn::operations::moreh::moreh_softmax::MorehLogSoftmax>();
}  // namespace ttnn
