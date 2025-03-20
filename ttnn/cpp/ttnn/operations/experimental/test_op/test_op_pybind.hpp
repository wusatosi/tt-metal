// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "pybind11/pybind_fwd.hpp"

namespace ttnn::operations::experimental::test_op::detail {
namespace py = pybind11;
void bind_experimental_test_op_operation(py::module& module);

}  // namespace ttnn::operations::experimental::test_op::detail
