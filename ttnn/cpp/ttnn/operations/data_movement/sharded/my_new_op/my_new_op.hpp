// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/core_coord.hpp>
#include "device/my_new_op_operation.hpp"

namespace ttnn {
namespace operations::data_movement {

struct MyNewOpOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor1, const ttnn::Tensor& input_tensor2, const int input_scalar);
};
}  // namespace operations::data_movement

constexpr auto my_new_op = ttnn::
    register_operation_with_auto_launch_op<"ttnn::my_new_op", operations::data_movement::MyNewOpDeviceOperation>();
}  // namespace ttnn
