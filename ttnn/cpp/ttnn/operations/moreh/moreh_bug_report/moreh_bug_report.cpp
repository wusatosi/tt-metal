// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bug_report.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/moreh/moreh_bug_report/device/moreh_bug_report_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_bug_report {
Tensor MorehBugReport::invoke(const Tensor& input) { return ttnn::prim::moreh_bug_report(input); }
}  // namespace ttnn::operations::moreh::moreh_bug_report
