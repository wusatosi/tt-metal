// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"
#include "serialization/serializable.hpp"

namespace ttml::ops {

typedef std::unordered_map<std::string, serialization::SerializableType> Parameters;

autograd::TensorPtr fused_gpt_block(const autograd::TensorPtr& input, Parameters& parameters);

}  // namespace ttml::ops
