// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm.hpp"

#include "device/batch_norm_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_mean/device/moreh_mean_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::normalization {

inline Tensor mean_NHW(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
    auto batch_mean = input_tensor;
    ttnn::SmallVector<int64_t> dims = {0, 2, 3};
    std::sort(dims.begin(), dims.end());
    for (uint32_t i = dims.size() - 1; i > 0; i--) {
        auto temp_output = ttnn::prim::moreh_mean(
            batch_mean, dims[i], true, std::nullopt, std::nullopt, output_memory_config, std::nullopt);
        batch_mean = temp_output;
    }
    return ttnn::prim::moreh_mean(
        batch_mean, dims.front(), true, std::nullopt, std::nullopt, output_memory_config, std::nullopt);
}

Tensor BatchNorm::invoke(
    const Tensor& input,
    std::optional<Tensor> running_mean,
    std::optional<Tensor> running_var,
    const bool training,
    const float eps,
    const float momentum,
    std::optional<Tensor> weight,
    std::optional<Tensor> bias,
    std::optional<Tensor> output,
    const std::optional<MemoryConfig>& memory_config) {
    std::cout << "HERE 1" << std::endl;
    if (training) {
        // Tensor batch_mean = mean_NHW(input, memory_config);  // calculated batch mean
        std::cout << "HERE 2" << std::endl;
        auto result = ttnn::prim::batch_norm(
            input,
            running_mean.value(),
            running_var.value(),
            eps,
            weight,
            bias,
            training,
            output,
            running_mean,
            memory_config);
        std::cout << "HERE 3" << std::endl;
        // running_mean.value() = batch_mean;
        std::cout << "HERE 4" << std::endl;
        return result;
    }
    TT_FATAL(
        (running_mean.has_value() && running_var.has_value()),
        "running_mean and running_var must be defined in evaluation mode");
    return ttnn::prim::batch_norm(
        input,
        running_mean.value(),
        running_var.value(),
        eps,
        weight,
        bias,
        training,
        output,
        std::nullopt,
        memory_config);
}
}  // namespace ttnn::operations::normalization
