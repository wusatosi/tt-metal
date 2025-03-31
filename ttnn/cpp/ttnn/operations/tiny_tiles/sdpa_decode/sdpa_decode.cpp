// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode.hpp"

#include <utility>

#include "device/sdpa_decode_op.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::tiny_tiles {

ttnn::Tensor ExecuteScaledDotProductAttentionDecode::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v) {
    return operation::run(
               ScaledDotProductAttentionDecode{}, {input_tensor_q, input_tensor_k, input_tensor_v}, {}, {}, queue_id)
        .at(0);
}

ttnn::Tensor ExecuteScaledDotProductAttentionDecode::invoke(
    const ttnn::Tensor& input_tensor_q, const ttnn::Tensor& input_tensor_k, const ttnn::Tensor& input_tensor_v) {
    return invoke(DefaultQueueId, input_tensor_q, input_tensor_k, input_tensor_v);
}

}  // namespace ttnn::operations::tiny_tiles
