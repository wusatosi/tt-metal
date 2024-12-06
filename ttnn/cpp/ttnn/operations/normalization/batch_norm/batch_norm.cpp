// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm.hpp"

#include "device/batch_norm_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_mean/device/moreh_mean_device_operation.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/unary/unary_composite.hpp"

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
    const Tensor& batch_mean,
    std::optional<Tensor> output,
    const std::optional<MemoryConfig>& memory_config) {
    // moreh mean code
    // Tensor batch_mean = mean_NHW(input, memory_config);
    // Tensor mean_sq = mean_NHW(ttnn::square(input, memory_config), memory_config);
    // Tensor batch_var = ttnn::subtract(mean_sq, ttnn::square(batch_mean, memory_config), std::nullopt, memory_config);

    // send mean as one input and check
    return ttnn::prim::batch_norm(input, batch_mean, output, memory_config);
}
}  // namespace ttnn::operations::normalization
