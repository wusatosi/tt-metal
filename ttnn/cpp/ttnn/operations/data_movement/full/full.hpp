// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

struct FullOperation {
    static ttnn::Tensor invoke(
        const uint8_t queue_id,
        const ttnn::SimpleShape& logical_shape,
        const float fill_value,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);
    static ttnn::Tensor invoke(
        const ttnn::SimpleShape& logical_shape,
        const float fill_value,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace data_movement
}  // namespace operations

constexpr auto full = ttnn::register_operation<"ttnn::full", ttnn::operations::data_movement::FullOperation>();

}  // namespace ttnn
