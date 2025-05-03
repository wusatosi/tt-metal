#pragma once

#include <array>
#include <vector>
#include <variant>
#include <optional>
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/common/queue_id.hpp"

namespace ttnn {
namespace operations {
namespace conv {
namespace conv2dmatmul {

struct conv2dMatmul {
    static Tensor invoke(
        const Tensor& input_tensor,
        uint32_t in_channels,
        uint32_t out_channels,
        uint32_t batch_size,
        uint32_t input_height,
        uint32_t input_width,
        std::array<uint32_t, 2> kernel_size,
        std::array<uint32_t, 2> stride);
};

}  // namespace conv2dmatmul
}  // namespace conv
}  // namespace operations

constexpr auto conv2dmatmul =
    ttnn::register_operation<"ttnn::conv2dmatmul", ttnn::operations::conv::conv2dmatmul::conv2dMatmul>();
}  // namespace ttnn
