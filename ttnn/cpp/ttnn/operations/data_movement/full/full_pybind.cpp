// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "full_pybind.hpp"
#include "full.hpp"
#include "cpp/pybind11/decorators.hpp"

namespace ttnn::operations::data_movement {
namespace detail {
namespace py = pybind11;

void bind_full_op(py::module& module) {
    auto doc = fmt::format(
        R"doc(
            Creates a tensor of the specified shape and fills it with the specified value.

            +---------------+-------------------------------------+-----------------------+------------------------+----------+
            | Argument      | Description                         | Data type             | Valid range            | Required |
            +===============+=====================================+=======================+========================+==========+
            | logical_shape | Shape of output tensor              | shape, tuple, etc     |                        | Yes      |
            +---------------+-------------------------------------+-----------------------+------------------------+----------+
            | fill_value    | value to fill into padding          | float                 | [-inf , inf]           | Yes      |
            +---------------+-------------------------------------+-----------------------+------------------------+----------+
            Args:
                logical_shape : Shape of the output tensor. Must be a tuple of integer values greater than 0
                fill_value (float): Value to fill the tensor with.
            Keyword args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to
                `None`. queue_id (int, optional): command queue id. Defaults to `0`.
            Returns:
                ttnn.Tensor: the output tensor.
        )doc",
        ttnn::full.base_name());

    using OperationType = decltype(ttnn::full);
    ttnn::bind_registered_operation(
        module,
        ttnn::full,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::SmallVector<int32_t> shape,
               const float fill_value,
               const std::optional<MemoryConfig>& memory_config,
               uint8_t queue_id) { return self(queue_id, shape, fill_value, memory_config); },
            py::arg("shape"),
            py::arg("fill_value"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = 0});
}

}  // namespace detail

void bind_full(py::module& module) { detail::bind_full_op(module); }

}  // namespace ttnn::operations::data_movement
