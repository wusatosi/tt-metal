// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

struct SliceOperation {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        tt::stl::Span<const uint32_t> begins,
        tt::stl::Span<const uint32_t> ends,
        tt::stl::Span<const uint32_t> step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        tt::stl::Span<const int> begins,
        tt::stl::Span<const int> ends,
        tt::stl::Span<const int> step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        tt::stl::Span<const uint32_t> output_tensor_start,
        tt::stl::Span<const uint32_t> output_tensor_end,
        tt::stl::Span<const uint32_t> step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        tt::stl::Span<const int> output_tensor_start,
        tt::stl::Span<const int> output_tensor_end,
        tt::stl::Span<const int> step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::SmallVector<uint32_t>& begins,
        const ttnn::SmallVector<uint32_t>& ends,
        const ttnn::SmallVector<uint32_t>& step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return invoke(
            queue_id,
            input_tensor,
            tt::stl::Span(begins),
            tt::stl::Span(ends),
            tt::stl::Span(step),
            memory_config_arg,
            optional_output_tensor);
    }
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::SmallVector<int>& begins,
        const ttnn::SmallVector<int>& ends,
        const ttnn::SmallVector<int>& step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return invoke(
            queue_id,
            input_tensor,
            tt::stl::Span(begins),
            tt::stl::Span(ends),
            tt::stl::Span(step),
            memory_config_arg,
            optional_output_tensor);
    }

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::SmallVector<uint32_t>& begins,
        const ttnn::SmallVector<uint32_t>& ends,
        const ttnn::SmallVector<uint32_t>& step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return invoke(
            input_tensor,
            tt::stl::Span(begins),
            tt::stl::Span(ends),
            tt::stl::Span(step),
            memory_config_arg,
            optional_output_tensor);
    }
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::SmallVector<int>& begins,
        const ttnn::SmallVector<int>& ends,
        const ttnn::SmallVector<int>& step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return invoke(
            input_tensor,
            tt::stl::Span(begins),
            tt::stl::Span(ends),
            tt::stl::Span(step),
            memory_config_arg,
            optional_output_tensor);
    }
};

}  // namespace data_movement
}  // namespace operations

constexpr auto slice =
    ttnn::register_operation_with_auto_launch_op<"ttnn::slice", ttnn::operations::data_movement::SliceOperation>();

}  // namespace ttnn
