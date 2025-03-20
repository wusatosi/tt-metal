// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/test_op_device_operation.hpp"
#include "test_op.hpp"

namespace ttnn::operations::experimental {

Tensor TestOperation::invoke(
    QueueId queue_id,
    const Tensor& inp0,
    const Tensor& inp1,
    const string& metainfo,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_out) {
    DataType output_dtype = inp0.get_dtype();
    auto arch = inp0.device()->arch();
    auto output_memory_config =
        optional_out.has_value() ? optional_out.value().memory_config() : memory_config.value_or(inp0.memory_config());

    return ttnn::prim::test_op(queue_id, inp0, inp1, metainfo, output_dtype, output_memory_config, optional_out);
}
}  // namespace ttnn::operations::experimental
