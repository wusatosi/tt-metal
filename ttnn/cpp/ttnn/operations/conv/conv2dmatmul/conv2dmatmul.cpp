// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/conv/conv2dmatmul/conv2dmatmul.hpp"
#include <optional>
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/view/view.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/conv/conv2dmatmul/device/conv2dmatmul_op.hpp"

#include "ttnn/operations/core/core.hpp"
namespace ttnn {
namespace operations {
namespace conv {
namespace conv2dmatmul {

Tensor conv2dMatmul::invoke(
    const Tensor& input_tensor,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride) {
    auto device = input_tensor.device();
    Tensor temp_in = ttnn::to_layout(input_tensor, Layout::ROW_MAJOR, std::nullopt, std::nullopt, device);
    auto temp_out = conv2d_convert_tensor_for_matmul(
        temp_in, in_channels, out_channels, batch_size, input_height, input_width, kernel_size, stride);
    Tensor in = ttnn::to_layout(temp_out, Layout::TILE, std::nullopt, std::nullopt, device);
    // in = ttnn::reshape(
    //     in,
    //     ttnn::Shape(
    //         {batch_size, input_height / stride[0], stride[0], input_width / stride[1], stride[1], in_channels}));
    // // in = ttnn::permute(in, ttnn::SmallVector<int64_t>({0, 1, 3, 2, 4, 5}));
    in = ttnn::reshape(
        in,
        ttnn::Shape(
            {batch_size, input_height / stride[0], input_width / stride[1], (in_channels)*stride[0] * stride[1]}));

    return in;
}

}  // namespace conv2dmatmul
}  // namespace conv
}  // namespace operations
}  // namespace ttnn
