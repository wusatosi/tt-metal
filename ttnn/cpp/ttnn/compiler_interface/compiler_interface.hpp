// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/compiler_interface/singleton_device_context.hpp"
#include "ttnn/cpp/ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::compiler_interface {

using OperandParams =
    std::tuple<ttnn::SimpleShape, tt::tt_metal::DataType, tt::tt_metal::Layout, tt::tt_metal::MemoryConfig>;

std::tuple<bool, size_t, size_t, size_t> extract_data_from_trace(const nlohmann::json& trace) {
    auto interleave_storage_cores = SingletonDeviceContext::get_instance().get_compute_with_storage_grid_size();

    size_t peak_cbs_per_core = graph::extract_circular_buffers_peak_size_per_core(trace);
    size_t peak_l1_tensors_per_core =
        graph::extract_l1_buffer_allocation_peak_size_per_core(trace, interleave_storage_cores);
    size_t output_l1_tensor_per_core =
        graph::extract_l1_output_buffer_allocation_size_per_core(trace, interleave_storage_cores);
    bool constraint_valid = true;

    return {constraint_valid, peak_cbs_per_core, peak_l1_tensors_per_core, output_l1_tensor_per_core};
}

}  // namespace ttnn::compiler_interface
