// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "tt_metal/impl/buffers/global_circular_buffer.hpp"
#include "tt_metal/include/tt_metal/global_circular_buffer.hpp"
#include "ttnn/cpp/ttnn/global_circular_buffer.hpp"
#include "device/dram_prefetcher_op.hpp"

namespace ttnn {
namespace operations::dram_prefetcher {

struct ExecuteDramPrefetcher {
    static ttnn::Tensor invoke(
        std::vector<ttnn::Tensor>& tensors,
        const uint32_t num_layers,
        const std::optional<const DeviceGlobalCircularBuffer>& global_cb);
};

}  // namespace operations::dram_prefetcher

constexpr auto dram_prefetcher = ttnn::register_operation_with_auto_launch_op<
    "ttnn::dram_prefetcher",
    ttnn::operations::dram_prefetcher::ExecuteDramPrefetcher>();

}  // namespace ttnn
