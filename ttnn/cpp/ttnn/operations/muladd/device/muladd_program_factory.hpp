// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt-metalium/base_types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::muladd {

operation::ProgramWithCallbacks single_core_muladd(
    const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, Tensor& output, MathFidelity math_fidelity);
}  // namespace ttnn::operations::muladd
