// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::operation::ProgramWithCallbacks my_new_op_multi_core(
    const Tensor& a,
    const Tensor& b,
    const Tensor& output,
    float scalar_multiplier = 1.0f,
    uint32_t num_slices = 1,
    uint32_t slice_index = 0);
}
