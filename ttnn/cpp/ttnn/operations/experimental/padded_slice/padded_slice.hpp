// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

struct PaddedSliceOperation {
    template <typename T>
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        tt::stl::Span<const T> begins,
        tt::stl::Span<const T> ends,
        tt::stl::Span<const T> step,
        const std::optional<MemoryConfig>& memory_config_arg,
        const std::optional<Tensor>& optional_output_tensor,
        const std::optional<float>& pad_value);

    template <typename T, std::size_t N>
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const std::array<T, N>& output_tensor_start,
        const std::array<T, N>& output_tensor_end,
        const std::array<T, N>& step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<float>& pad_value = std::nullopt);
};

}  // namespace data_movement
}  // namespace operations

constexpr auto padded_slice = ttnn::register_operation_with_auto_launch_op<
    "ttnn::padded_slice",
    ttnn::operations::data_movement::PaddedSliceOperation>();

}  // namespace ttnn
