// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
namespace ttnn::operations::moreh::moreh_bug_report {
struct MorehBugReport {
    static Tensor invoke(const Tensor& input);
};
}  // namespace ttnn::operations::moreh::moreh_bug_report

namespace ttnn {
constexpr auto moreh_bug_report = ttnn::register_operation_with_auto_launch_op<
    "ttnn::moreh_bug_report",
    ttnn::operations::moreh::moreh_bug_report::MorehBugReport>();
}
