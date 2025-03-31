// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::tiny_tiles::detail {

tt::tt_metal::operation::ProgramWithCallbacks sdpa_decode_multi_core(
    const Tensor& input_tensor_q, const Tensor& input_tensor_k, const Tensor& input_tensor_v);

}  // namespace ttnn::operations::tiny_tiles::detail
