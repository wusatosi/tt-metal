// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <stdio.h>
#include "stack.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/concat/concat.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor StackOperation::invoke(const std::vector<ttnn::Tensor>& input_tensors, const int dim) {
    std::cout << "Inside custom implmentation of stack\n";
    if (input_tensors.empty()) {
        throw std::invalid_argument("Input tensors list is empty.");
    }
    std::cout << "Calling ttnn::unsqueeze op\n";
    std::vector<ttnn::Tensor> expanded_tensors;
    for (const auto& tensor : input_tensors) {
        expanded_tensors.push_back(ttnn::unsqueeze(tensor, dim));
    }
    return ttnn::concat(expanded_tensors, dim);
}

}  // namespace ttnn::operations::data_movement
