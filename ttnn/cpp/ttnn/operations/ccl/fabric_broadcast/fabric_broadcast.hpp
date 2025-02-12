// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/run_operation.hpp"

#include "device/topk_op.hpp"
#include "ttnn/types.hpp"

template <class Tuple, class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::vector<std::optional<T>> tuple_to_vector_optional(Tuple&& tuple) {
    return std::apply(
        [](auto&&... elems) { return std::vector<std::optional<T>>{std::forward<decltype(elems)>(elems)...}; },
        std::forward<Tuple>(tuple));
}
namespace ttnn {
namespace operations::data_movement {

struct ExecuteFabricBroadcast {
    static inline Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        const uint16_t dest_device_id,
        const uint16_t dest_mesh_id,
        const std::optional<MemoryConfig>& memory_config) {
        return operation::run(
            FabricBroadcast{dest_device_id, dest_mesh_id, memory_config.value_or(input_tensor.memory_config())},
            {input_tensor},
            {},
            {},
            queue_id);
    }

    static inline auto invoke(
        const Tensor& input_tensor,
        const uint16_t dest_device_id,
        const uint16_t dest_mesh_id,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<std::tuple<Tensor, Tensor>> optional_output_tensors) {
        return invoke(DefaultQueueId, input_tensor, k, dim, largest, sorted, memory_config, optional_output_tensors);
    }

    static inline std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {
            Tensor(operation::get_workers_for_op_output({input_tensor})),
            Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }
};

}  // namespace operations::data_movement

constexpr auto topk =
    ttnn::register_operation_with_auto_launch_op<"ttnn::topk", ttnn::operations::data_movement::ExecuteTopK>();

}  // namespace ttnn
