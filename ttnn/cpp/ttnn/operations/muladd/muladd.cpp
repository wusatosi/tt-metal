#include "muladd.hpp"
#include <optional>
#include "device/muladd_op.hpp"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn {

namespace operations {
namespace muladd {

Tensor MulAddOperation::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const Tensor& input_tensor_c,
    const Tensor& input_tensor_d,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DataType> dtype,
    const std::optional<MathFidelity> math_fidelity) {
    auto program = MulAdd{
        .memory_config = memory_config.value_or(input_tensor_a.memory_config()),
        .dtype = dtype.value_or(input_tensor_a.dtype()),
        .math_fidelity = math_fidelity.value_or(MathFidelity::HiFi4)};

    return operation::run(program, {input_tensor_a, input_tensor_b, input_tensor_c, input_tensor_d}, {}, {}, queue_id)
        .at(0);
}
}  // namespace muladd
}  // namespace operations
}  // namespace ttnn
