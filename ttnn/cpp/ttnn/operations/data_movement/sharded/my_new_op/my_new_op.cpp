// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include "device/my_new_op_operation.hpp"
#include "my_new_op.hpp"
#include <tt-metalium/work_split.hpp>
#include <magic_enum/magic_enum.hpp>
#include <utility>
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "cpp/ttnn/operations/eltwise/ternary/where.hpp"
#include "cpp/ttnn/operations/copy.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/creation.hpp"
#include "cpp/ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor MyNewOpOperation::invoke(
    const ttnn::Tensor& input_tensor1, const ttnn::Tensor& input_tensor2, const int input_scalar) {
    return ttnn::my_new_op(input_tensor1, input_tensor2, input_scalar);
}
}  // namespace ttnn::operations::data_movement
