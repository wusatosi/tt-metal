// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm.hpp"

#include "device/batch_norm_device_operation.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::normalization {

inline Tensor reshape_to_4D(const ttnn::SimpleShape& input_tensor_shape, const std::optional<Tensor> reshaping_tensor) {
    // auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
    // const auto in_shape = input_tensor.get_logical_shape();
    const auto stat_shape = reshaping_tensor.value().get_logical_shape();
    TT_FATAL(
        stat_shape[-1] == input_tensor_shape[1],
        "Mismatch in channel size. Found {} instead of channel size = {}.",
        stat_shape[-1],
        input_tensor_shape[1]);
    Tensor b = reshaping_tensor.value();
    if (stat_shape.rank() < 3) {
        b = ttnn::reshape(
            reshaping_tensor.value(), ttnn::SimpleShape(std::array<uint32_t, 4>{1, input_tensor_shape[1], 1, 1}));
    }
    return b;
}

Tensor BatchNorm::invoke(
    const Tensor& input,
    std::optional<Tensor> running_mean,
    std::optional<Tensor> running_var,
    const bool training,
    const float eps,
    std::optional<Tensor> weight,
    std::optional<Tensor> bias,
    std::optional<Tensor> output,
    const std::optional<MemoryConfig>& memory_config) {
    const auto in_shape = input.get_logical_shape();
    if (running_mean.has_value()) {
        running_mean = reshape_to_4D(in_shape, running_mean.value());
    }
    if (running_var.has_value()) {
        running_var = reshape_to_4D(in_shape, running_var.value());
    }
    if (weight.has_value()) {
        weight = reshape_to_4D(in_shape, weight.value());
    }
    if (bias.has_value()) {
        bias = reshape_to_4D(in_shape, bias.value());
    }
    // TODO: Implementation for training mode is in progress
    TT_FATAL((!training), "Support currently provided for inference mode only");
    TT_FATAL(
        (running_mean.has_value() && running_var.has_value()),
        "running_mean and running_var must be defined in evaluation mode");
    return ttnn::prim::batch_norm(
        input, running_mean.value(), running_var.value(), eps, weight, bias, output, memory_config);
}
}  // namespace ttnn::operations::normalization
