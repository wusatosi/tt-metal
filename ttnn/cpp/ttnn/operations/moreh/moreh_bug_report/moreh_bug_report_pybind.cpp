// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bug_report_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "ttnn/operations/moreh/moreh_bug_report/moreh_bug_report.hpp"

namespace ttnn::operations::moreh::moreh_bug_report {
void bind_moreh_bug_report_operation(py::module& module) {
    bind_registered_operation(
        module, ttnn::moreh_bug_report, "Moreh bug_report Operation", ttnn::pybind_arguments_t{py::arg("input")});
}
}  // namespace ttnn::operations::moreh::moreh_bug_report
