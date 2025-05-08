// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sliced_reduce_scatter_async.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/sliced_reduce_scatter/device/sliced_reduce_scatter_async_op.hpp"
#include "ttnn/distributed/types.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteSlicedReduceScatterAsync::invoke(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& persistent_intermediate_buffer,
    ttnn::Tensor& persistent_output_buffer,
    const int32_t scatter_dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    return ttnn::operations::experimental::ccl::sliced_reduce_scatter_async(
        input_tensor,
        persistent_intermediate_buffer,
        persistent_output_buffer,
        scatter_dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology,
        subdevice_id);
}

}  // namespace ttnn::operations::experimental::ccl
