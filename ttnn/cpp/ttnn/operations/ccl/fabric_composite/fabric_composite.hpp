// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"

namespace ttnn {

namespace operations::fabric_composite {

struct ExecuteFabricBroadcast {
    static Tensor invoke(QueueId queue_id, const Tensor& input, distributed::MeshDevice* mesh_device) {
        TT_FATAL(input.storage_type() == tt::tt_metal::StorageType::DEVICE, "Input tensor must be on device");
        std::vector<Tensor> tensors;
        tensors.reserve(mesh_device->num_devices());
        for (auto* device : mesh_device->get_devices()) {
            if (input.device()->id() != device->id()) {
                tensors.push_back(ttnn::unicast(queue_id, input, mesh_device, device->id()));
            } else {
                tensors.push_back(input);
            }
        }
        return distributed::create_multi_device_tensor(
            tensors, tt::tt_metal::StorageType::MULTI_DEVICE, tt::tt_metal::ReplicateTensor());
    }
};

struct ExecuteFabricScatter {
    static Tensor invoke(QueueId queue_id, const Tensor& input, distributed::MeshDevice* mesh_device, int dim) {
        TT_FATAL(input.storage_type() == tt::tt_metal::StorageType::DEVICE, "Input tensor must be on device");

        const auto num_devices = mesh_device->num_devices();

        const auto tensor_shape = input.get_logical_shape();
        const auto dim_size = tensor_shape[dim];
        TT_FATAL(dim_size % num_devices != 0, "Dimension size must be divisible by number of devices");
        auto split_size_per_device = dim_size / num_devices;

        ttnn::SmallVector<uint32_t> start{0, 0, 0, 0};
        ttnn::SmallVector<uint32_t> end{tensor_shape[0], tensor_shape[1], tensor_shape[2], tensor_shape[3]};
        ttnn::SmallVector<uint32_t> stride{1U, 1U, 1U, 1U};

        std::vector<tt::tt_metal::Tensor> scattered_tensors;
        scattered_tensors.reserve(num_devices);
        for (size_t device_index = 0; device_index < num_devices; ++device_index) {
            auto* device = mesh_device->get_device(device_index);

            start[dim] = split_size_per_device * device_index;
            end[dim] = split_size_per_device * (device_index + 1);

            auto sliced_tensor = ttnn::slice(input, start, end, stride);
            if (input.device()->id() != device->id()) {
                scattered_tensors.push_back(ttnn::unicast(queue_id, sliced_tensor, mesh_device, device->id()));
            } else {
                scattered_tensors.push_back(sliced_tensor);
            }
        }
        return distributed::create_multi_device_tensor(
            scattered_tensors, tt::tt_metal::StorageType::MULTI_DEVICE, tt::tt_metal::ReplicateTensor());
    }
};

}  // namespace operations::fabric_composite

constexpr auto fabric_broadcast =
    ttnn::register_operation<"ttnn::fabric_broadcast", operations::fabric_composite::ExecuteFabricBroadcast>();
constexpr auto fabric_scatter =
    ttnn::register_operation<"ttnn::fabric_scatter", operations::fabric_composite::ExecuteFabricScatter>();

}  // namespace ttnn
