// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"


namespace ttnn {
namespace operations::data_movement {
namespace detail {
    ttnn::Tensor host_reshape(const ttnn::Tensor& tensor, const ttnn::Shape& shape);
    ttnn::Tensor convert_tensor_to_rm_reshape_convert_back_to_orig_layout(const ttnn::Tensor& tensor, const ttnn::Shape& shape);
    ttnn::Tensor fix_shape_and_perform_reshape_on_2D_RM(const ttnn::Tensor& tensor, const ttnn::Shape& shape);
    ttnn::Tensor fix_shape_and_perform_reshape_on_3D_TILE(const ttnn::Tensor& tensor, const ttnn::Shape& shape);
    ttnn::Tensor perform_reshape_on_2D_RM(const ttnn::Tensor& tensor, const ttnn::Shape& shape);
    ttnn::Tensor perform_reshape_on_3D_TILE(const ttnn::Tensor& tensor, const ttnn::Shape& shape);

}

ttnn::Shape tiling_reshape_corrector(const ttnn::Shape& shape);
ttnn::Tensor PerformView(const ttnn::Tensor& tensor, const ttnn::Shape& shape);
void Validate_transform (const ttnn::Shape& input_shape, const ttnn::Shape& output_shape);

struct ReshapeViewOperation {
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, const ttnn::Shape& shape);
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, const ttnn::SimpleShape& logical_shape);
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, tt::stl::Span<const int32_t> shape_vector);
};


}  // namespace operations::data_movement

constexpr auto reshape = ttnn::register_operation<"ttnn::reshape", ttnn::operations::data_movement::ReshapeViewOperation>();

}  // namespace ttnn
