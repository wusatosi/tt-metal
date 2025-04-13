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
    auto t = ttnn::prim::deinterleave_to_batch(
        input, input_height, input_width, stride_hw, barrier_threshold, compute_kernel_config);
    return t;
}

std::vector<Tensor> DeinterleaveLocal::invoke(
    const Tensor& input,
    const uint32_t input_height,
    const uint32_t input_width,
    const std::array<uint32_t, 2> stride_hw,
    const uint32_t barrier_threshold,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::deinterleave_local(
        input, input_height, input_width, stride_hw, barrier_threshold, compute_kernel_config);
}

std::vector<Tensor> DeinterleaveLocal::create_async_output_tensors(
    const Tensor& input,
    const uint32_t input_height,
    const uint32_t input_width,
    const std::array<uint32_t, 2> stride_hw,
    const uint32_t barrier_threshold,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    tt::log_warning(tt::LogOp, "DeinterleaveLocal::create_async_output_tensors");

    return ttnn::operations::experimental::deinterleave::DeinterleaveLocalOperation::create_output_tensors(
        {
            input_height,
            input_width,
            stride_hw,
            barrier_threshold,
            init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4),
        },
        {input});
}

}  // namespace ttnn::operations::experimental::deinterleave
