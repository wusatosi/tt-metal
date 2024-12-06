// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_pybind.hpp"

#include "batch_norm.hpp"

#include "pybind11/decorators.hpp"
namespace py = pybind11;
namespace ttnn::operations::normalization::detail {
void bind_batch_norm_operation(pybind11::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::batch_norm,
        "batch_norm Operation",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("batch_mean"),
            py::kw_only(),
            py::arg("output") = std::nullopt,
            py::arg("memory_config") = std::nullopt

        });
}
}  // namespace ttnn::operations::normalization::detail
