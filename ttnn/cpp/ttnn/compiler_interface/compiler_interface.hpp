// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/cpp/ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::compiler_interface {

using OperandParams =
    std::tuple<ttnn::SimpleShape, tt::tt_metal::DataType, tt::tt_metal::Layout, tt::tt_metal::MemoryConfig>;

}  // namespace ttnn::compiler_interface
