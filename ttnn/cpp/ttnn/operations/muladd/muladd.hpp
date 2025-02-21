#pragma once
#include <ttnn/decorators.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_utils.hpp>
#include "ttnn/tensor/types.hpp"
namespace ttnn {
namespace operations {
namespace muladd {

// ttnn:Tensor bound_muladd(
//     QueueId queue_id,
//     const ttnn::Tensor& input_tensor_a,
//     const ttnn::Tensor& input_tensor_b,
//     const ttnn::Tensor& input_tensor_c,
//     const ttnn::Tensor& input_tensor_d,
// );

struct MulAddOperation {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const Tensor& input_tensor_c,
        const Tensor& input_tensor_d,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<DataType> dtype = std::nullopt,
        const std::optional<MathFidelity> math_fidelity = std::nullopt);
};

}  // namespace muladd
}  // namespace operations
constexpr auto muladd = ttnn::register_operation<"ttnn::muladd", operations::muladd::MulAddOperation>();
}  // namespace ttnn
