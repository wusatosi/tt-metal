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
    const tt::tt_metal::MemoryConfig& distributed_mem_config;
    const int block_size;
    const int num_blocks_per_core;
    const int num_cores_with_extra_block;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "distributed_mem_config", "block_size", "num_blocks_per_core", "num_cores_with_extra_block");
    const auto attribute_values() const {
        return std::make_tuple(
            this->distributed_mem_config,
            this->block_size,
            this->num_blocks_per_core,
            this->num_cores_with_extra_block);
    }
};
}  // namespace ttnn::operations::data_movement
