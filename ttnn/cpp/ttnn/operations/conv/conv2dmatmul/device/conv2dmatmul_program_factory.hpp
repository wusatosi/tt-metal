// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt-metalium/host_api.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::conv::conv2dmatmul::detail {

tt::tt_metal::operation::ProgramWithCallbacks conv2dmatmul_tile_reader_writer(
    const Tensor& input_tensor,
    const Tensor& output,
    const uint32_t in_channels,
    const uint32_t out_channels,
    const uint32_t batch_size,
    const uint32_t input_height,
    const uint32_t input_width,
    const std::array<uint32_t, 2> kernel_size,
    const std::array<uint32_t, 2> stride);

}  // namespace ttnn::operations::conv::conv2dmatmul::detail
