// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct RepeatOperation {
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, const ttnn::SmallVector<uint32_t>& repetition_vector);
};

}  // namespace operations::data_movement

constexpr auto repeat = ttnn::register_operation<"ttnn::repeat", ttnn::operations::data_movement::RepeatOperation>();

}  // namespace ttnn
