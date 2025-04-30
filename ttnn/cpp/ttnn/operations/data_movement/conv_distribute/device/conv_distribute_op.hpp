// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement {

struct ConvDistributeDeviceOperation {
    const ttnn::CoreRangeSet& cores;
    const ttnn::SmallVector<size_t>& shard_sizes;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple("cores", "shard_sizes");
    const auto attribute_values() const { return std::make_tuple(this->cores, this->shard_sizes); }
};
}  // namespace ttnn::operations::data_movement
