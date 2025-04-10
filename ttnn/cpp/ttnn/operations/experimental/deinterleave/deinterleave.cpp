// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deinterleave.hpp"

#include "device/deinterleave_device_operation.hpp"

namespace ttnn::operations::experimental::deinterleave {

Tensor DeinterleaveToBatch::invoke(
    const Tensor& input,
    const uint32_t input_height,
    const uint32_t input_width,
    const std::array<uint32_t, 2> stride_hw,
    const uint32_t barrier_threshold,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::deinterleave(
        input, input_height, input_width, stride_hw, true, barrier_threshold, compute_kernel_config);
}

Tensor DeinterleaveLocal::invoke(
    const Tensor& input,
    const uint32_t input_height,
    const uint32_t input_width,
    const std::array<uint32_t, 2> stride_hw,
    const uint32_t barrier_threshold,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::deinterleave(
        input, input_height, input_width, stride_hw, false, barrier_threshold, compute_kernel_config);
}
}  // namespace ttnn::operations::experimental::deinterleave
