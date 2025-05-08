// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/experimental/ccl/all_gather_matmul/device/all_gather_matmul_op.hpp"
#include "cpp/ttnn/distributed/api.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteSample {
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto sample =
    ttnn::register_operation<"ttnn::experimental::sample", ttnn::operations::experimental::ccl::ExecuteSample>();

}  // namespace experimental
}  // namespace ttnn
