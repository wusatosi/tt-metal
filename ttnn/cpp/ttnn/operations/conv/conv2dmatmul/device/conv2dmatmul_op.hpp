// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::conv {
namespace conv2dmatmul {
struct Conv2dMatmulOp {
    const uint32_t in_channels;
    const uint32_t out_channels;
    const uint32_t batch_size;
    const uint32_t input_height;
    const uint32_t input_width;
    const std::array<uint32_t, 2> kernel_size;
    const std::array<uint32_t, 2> stride;
    const tt::tt_metal::MemoryConfig output_mem_config;

    void validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;

    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;

    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "in_channels",
        "out_channels",
        "batch_size",
        "input_height",
        "input_width",
        "kernel_size",
        "stride",
        "output_mem_config");

    const auto attribute_values() const {
        return std::forward_as_tuple(
            this->in_channels,
            this->out_channels,
            this->batch_size,
            this->input_height,
            this->input_width,
            this->kernel_size,
            this->stride,
            this->output_mem_config);
    }
};

Tensor conv2d_convert_tensor_for_matmul(
    const Tensor& input_tensor,
    const uint32_t in_channels,
    const uint32_t out_channels,
    const uint32_t batch_size,
    const uint32_t input_height,
    const uint32_t input_width,
    const std::array<uint32_t, 2> kernel_size,
    const std::array<uint32_t, 2> stride);

}  // namespace conv2dmatmul
}  // namespace ttnn::operations::conv
