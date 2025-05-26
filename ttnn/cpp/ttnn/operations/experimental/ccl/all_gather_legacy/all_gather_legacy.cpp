// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_legacy.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/all_gather_legacy/device/all_gather_legacy_op.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteAllGatherLegacy::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& buffer_tensor,
    const int32_t dim,
    const GlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    return ttnn::operations::experimental::ccl::all_gather_legacy(
        input_tensor,
        buffer_tensor,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology,
        subdevice_id);
}

std::vector<ttnn::Tensor> ExecuteAllGatherLegacy::invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    const std::vector<ttnn::Tensor>& buffer_tensors,
    const int32_t dim,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    return ttnn::operations::experimental::ccl::all_gather_legacy(
        input_tensors,
        buffer_tensors,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology,
        subdevice_id);
}

}  // namespace ttnn::operations::experimental::ccl
