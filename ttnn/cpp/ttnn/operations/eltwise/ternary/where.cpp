// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "where.hpp"

#include <functional>
#include <utility>
#include <variant>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn {
namespace operations {
namespace ternary {

namespace ternary_utils {

// where - ternary operator y = (predicate) ? value_true : value_false; elementwise
// y = (predicate >= 0)*value_true + (predicate < 0)*value_false

using FloatOrTensor = std::variant<Tensor, float>;

Tensor where_impl(
    QueueId queue_id,
    const Tensor& predicate,
    const FloatOrTensor& value_true,
    const FloatOrTensor& value_false,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    auto get_multiplied = [&](const Tensor& condition, const FloatOrTensor& value) -> Tensor {
        if (std::holds_alternative<Tensor>(value)) {
            condition.print();
            std::get<Tensor>(value).print();
            return ttnn::multiply(queue_id, condition, std::get<Tensor>(value), std::nullopt, output_mem_config);
        } else {
            return ttnn::multiply(queue_id, condition, std::get<float>(value), std::nullopt, output_mem_config);
        }
    };
    predicate.print();
    if (std::holds_alternative<Tensor>(value_true)) {
        std::get<Tensor>(value_true).print();
    }
    if (std::holds_alternative<Tensor>(value_false)) {
        std::get<Tensor>(value_false).print();
    }
    Tensor t2 = get_multiplied(ttnn::gtz(queue_id, predicate, output_mem_config), value_true);
    t2.print();
    Tensor t1 = get_multiplied(ttnn::lez(queue_id, predicate, output_mem_config), value_false);
    t1.print();
    if (output_tensor.has_value()) {
        ttnn::add(queue_id, t2, t1, std::nullopt, output_mem_config, output_tensor);
    } else {
        output_tensor = ttnn::add(queue_id, t2, t1, std::nullopt, output_mem_config);
    }

    return output_tensor.value();
}

}  // namespace ternary_utils
Tensor WhereOperation::invoke(
    QueueId queue_id,
    const Tensor& predicate,
    const Tensor& value_true,
    const Tensor& value_false,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return ternary_utils::where_impl(
        queue_id,
        predicate,
        value_true,
        value_false,
        output_mem_config.value_or(predicate.memory_config()),
        std::move(output_tensor));
}

Tensor WhereOperation::invoke(
    QueueId queue_id,
    const Tensor& predicate,
    const float value_true,
    const Tensor& value_false,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return ternary_utils::where_impl(
        queue_id,
        predicate,
        value_true,
        value_false,
        output_mem_config.value_or(predicate.memory_config()),
        std::move(output_tensor));
}

Tensor WhereOperation::invoke(
    QueueId queue_id,
    const Tensor& predicate,
    const Tensor& value_true,
    const float value_false,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return ternary_utils::where_impl(
        queue_id,
        predicate,
        value_true,
        value_false,
        output_mem_config.value_or(predicate.memory_config()),
        std::move(output_tensor));
}

Tensor WhereOperation::invoke(
    QueueId queue_id,
    const Tensor& predicate,
    const float value_true,
    const float value_false,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return ternary_utils::where_impl(
        queue_id,
        predicate,
        value_true,
        value_false,
        output_mem_config.value_or(predicate.memory_config()),
        std::move(output_tensor));
}

}  // namespace ternary
}  // namespace operations
}  // namespace ttnn
