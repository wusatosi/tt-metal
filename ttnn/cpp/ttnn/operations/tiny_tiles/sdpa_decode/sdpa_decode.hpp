// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::tiny_tiles {

struct ExecuteScaledDotProductAttentionDecode {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& input_tensor_v);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor_q, const ttnn::Tensor& input_tensor_k, const ttnn::Tensor& input_tensor_v);
};

}  // namespace operations::tiny_tiles

namespace tiny_tiles {

constexpr auto sdpa_decode = ttnn::register_operation_with_auto_launch_op<
    "ttnn::tiny_tiles::sdpa_decode",
    ttnn::operations::tiny_tiles::ExecuteScaledDotProductAttentionDecode>();

}  // namespace tiny_tiles

}  // namespace ttnn
