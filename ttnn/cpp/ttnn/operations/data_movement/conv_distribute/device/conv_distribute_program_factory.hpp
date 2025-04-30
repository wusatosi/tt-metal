// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::operation::ProgramWithCallbacks conv_distribute_multi_core(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::CoreRangeSet& cores,
    const ttnn::SmallVector<size_t>& shard_sizes);
}
