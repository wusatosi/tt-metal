// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/run_operation.hpp"

#include "device/fabric_unicast_device_operation.hpp"
#include "ttnn/types.hpp"

template <class Tuple, class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::vector<std::optional<T>> tuple_to_vector_optional(Tuple&& tuple) {
    return std::apply(
        [](auto&&... elems) { return std::vector<std::optional<T>>{std::forward<decltype(elems)>(elems)...}; },
        std::forward<Tuple>(tuple));
}
namespace ttnn {
namespace operations::ccl {

struct ExecuteFabricUnicast {
    static inline Tensor invoke(
        QueueId queue_id, const Tensor& input_tensor, MeshDevice* mesh_device, const uint16_t dest_device_id) {
        return operation::run(
            FabricUnicast{MeshDevice * mesh_device, dest_device_id}, {input_tensor}, {}, {}, queue_id);
    }

    static inline auto invoke(
        const Tensor& input_tensor,
        const uint16_t dest_device_id,
        MeshDevice* mesh_device,
        const uint16_t dest_device_id) {
        return invoke(DefaultQueueId, input_tensor, mesh_device, device_id);
    }
};

}  // namespace operations::ccl

constexpr auto fabric_unicast =
    ttnn::register_operation_with_auto_launch_op<"ttnn::fabric_unicast", ttnn::operations::ccl::FabricUnicast>();

}  // namespace ttnn
