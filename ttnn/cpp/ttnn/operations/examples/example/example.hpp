
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/example_device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include <vector>

namespace ttnn::operations::examples {

// A composite operation is an operation that calls multiple operations in sequence
// It is written using invoke and can be used to call multiple primitive and/or composite operations
struct CompositeExampleOperation {
    // The user will be able to call this method as `Tensor output = ttnn::composite_example(input_tensor)` after the op
    // is registered
    static Tensor invoke(const Tensor& input_tensor, const Tensor& output_tensor) {
        auto copy = prim::example(input_tensor, output_tensor);
        auto another_copy = prim::example(copy, output_tensor);
        return another_copy;
    }
};

}  // namespace ttnn::operations::examples

namespace ttnn {
constexpr auto composite_example =
    ttnn::register_operation<"ttnn::composite_example", operations::examples::CompositeExampleOperation>();
}  // namespace ttnn
