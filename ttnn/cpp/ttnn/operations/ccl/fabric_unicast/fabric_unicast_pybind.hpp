// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"

#include "topk.hpp"
#include <optional>

namespace ttnn::operations::ccl::detail {
namespace py = pybind11;

void bind_fabric_unicast_operation(py::module& module) {
    auto doc =
        R"doc(
        )doc";

    using OperationType = decltype(ttnn::fabric_unicast);
    bind_registered_operation(
        module,
        ttnn::topk,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               MeshDevice* mesh_device,
               const uint16_t dest_device_id,
               QueueId queue_id) { return self(queue_id, input_tensor, mesh_device, dest_device_id); },
            py::arg("input_tensor").noconvert(),
            py::arg("mesh_device").no_convert(),
            py::arg("dest_device_id") = 0,
            py::kw_only(),
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::ccl::detail
