// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"

#include <iostream>
#include <tt-metalium/constants.hpp>

#include "sample.hpp"
#include "device/sample_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>

namespace ttnn::operations::experimental::ccl {
namespace detail {}  // namespace detail

ttnn::Tensor ExecuteSample::invoke(const ttnn::Tensor& input_tensor) {
    return ttnn::operations::experimental::ccl::sample(input_tensor);
}

}  // namespace ttnn::operations::experimental::ccl
