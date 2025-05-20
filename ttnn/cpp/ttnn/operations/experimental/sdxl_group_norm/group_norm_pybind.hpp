// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::experimental::sdxl_group_norm::detail {
namespace py = pybind11;
void bind_experimental_group_norm_operation(py::module& module);

}  // namespace ttnn::operations::experimental::sdxl_group_norm::detail
