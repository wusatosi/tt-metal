// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental {

struct TestOperation {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& inp0,
        const Tensor& inp1,
        const string& metainfo,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_out = std::nullopt);
};

}  // namespace ttnn::operations::experimental

namespace ttnn::experimental {
constexpr auto test_op = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::test_op",
    ttnn::operations::experimental::TestOperation>();
}
