
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/example_device_operation.hpp"

namespace ttnn::operations::examples {

struct NocInlineOperation {
    static Tensor invoke(const Tensor& input_tensor) {
        auto copy = ttnn::prim::noc_inline(input_tensor);
        return copy;
    }
};

}  // namespace ttnn::operations::examples

namespace ttnn {
constexpr auto inline_example =
    ttnn::register_operation<"ttnn::noc_inline", operations::examples::NocInlineOperation>();
}  // namespace ttnn
