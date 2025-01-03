
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/example_device_operation.hpp"

namespace ttnn::operations::examples {

// A composite operation is an operation that calls multiple operations in sequence
// It is written using invoke and can be used to call multiple primitive and/or composite operations
struct CompositeExampleOperation {
    // The user will be able to call this method as `Tensor output = ttnn::composite_example(input_tensor)` after the op
    // is registered
    static Tensor invoke(const Tensor& input_tensor, uint32_t block_size) {
        auto copy = prim::example(input_tensor, block_size);
        auto another_copy = prim::example(copy, block_size);
        return another_copy;
    }
};

}  // namespace ttnn::operations::examples

namespace ttnn {
constexpr auto composite_example =
    ttnn::register_operation<"ttnn::composite_example", operations::examples::CompositeExampleOperation>();
}  // namespace ttnn
