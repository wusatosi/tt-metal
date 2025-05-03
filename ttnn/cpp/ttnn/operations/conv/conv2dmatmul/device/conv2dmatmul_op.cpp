// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/conv/conv2dmatmul/device/conv2dmatmul_op.hpp"
#include "ttnn/operations/conv/conv2dmatmul/device/conv2dmatmul_program_factory.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::conv {
namespace conv2dmatmul {

void Conv2dMatmulOp::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_ASSERT(input_tensors.size() == 1, "Conv2dMatmul expects 1 input tensor");
    const auto& input_tensor = input_tensors[0];

    // Validate input tensor layout
    // TT_ASSERT(input_tensor.get_layout() == tt::tt_metal::Layout::TILE, "Input tensor must be in tile layout");
    TT_ASSERT(input_tensor.memory_config().is_dram(), "Input tensor must be DRAM");
}

std::vector<ttnn::TensorSpec> Conv2dMatmulOp::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];

    // Calculate output dimensions
    uint32_t output_height = input_height;
    uint32_t output_width = input_width;

    // Create output shape
    ttnn::Shape output_logical_shape({batch_size, output_height, output_width, in_channels});

    return {TensorSpec(
        output_logical_shape,
        tt::tt_metal::TensorLayout(
            input_tensor.get_dtype(),
            tt::tt_metal::PageConfig(input_tensor.get_layout()),
            input_tensor.memory_config()))};
}

tt::tt_metal::operation::ProgramWithCallbacks Conv2dMatmulOp::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto& output_tensor = output_tensors[0];

    // if (input_tensor.get_layout() == tt::tt_metal::Layout::ROW_MAJOR) {
    return detail::conv2dmatmul_tile_reader_writer(
        input_tensor,
        output_tensor,
        in_channels,
        out_channels,
        batch_size,
        input_height,
        input_width,
        kernel_size,
        stride);
    // } else {
    //     TT_THROW("Conv2dMatmul only supports tile layout");
    // }
}

Tensor conv2d_convert_tensor_for_matmul(
    const Tensor& input_tensor,
    const uint32_t in_channels,
    const uint32_t out_channels,
    const uint32_t batch_size,
    const uint32_t input_height,
    const uint32_t input_width,
    const std::array<uint32_t, 2> kernel_size,
    const std::array<uint32_t, 2> stride) {
    auto shape = input_tensor.padded_shape();
    tt::log_info("input_tensor shape = {} ", shape);
    std::vector<ttnn::Tensor> output_tensors = {
        ttnn::Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor}))};
    tt::tt_metal::operation::launch_op(
        [input_tensor, in_channels, out_channels, batch_size, input_height, input_width, kernel_size, stride](
            const std::vector<ttnn::Tensor>& input_tensors,
            const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
            const std::vector<std::optional<ttnn::Tensor>>& optional_output_tensors) mutable
            -> std::vector<ttnn::Tensor> {
            auto& a = input_tensors.at(0);
            auto conv2d_matmul =
                Conv2dMatmulOp(in_channels, out_channels, batch_size, input_height, input_width, kernel_size, stride);
            return tt::tt_metal::operation::run_without_autoformat(
                conv2d_matmul, input_tensors, optional_input_tensors);
        },
        {input_tensor},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace conv2dmatmul
}  // namespace ttnn::operations::conv
