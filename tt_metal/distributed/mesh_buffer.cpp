// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/mesh_buffer.hpp"
#include "tt_metal/detail/tt_metal.hpp"

namespace tt::tt_metal::distributed {

std::shared_ptr<MeshBuffer> MeshBuffer::create(const MeshBufferConfig& config, std::optional<DeviceAddr> address) {
    if (address.has_value()) {
        return std::shared_ptr<MeshBuffer>(new MeshBuffer(config, address));
    } else {
        return std::shared_ptr<MeshBuffer>(new MeshBuffer(config, address), deleter);
    }
}

MeshBuffer::MeshBuffer(const MeshBufferConfig& mesh_buffer_config, std::optional<DeviceAddr> address) {
    std::visit(  
        tt::stl::overloaded{
            [&](const ReplicatedBufferConfig& config) {
                mesh_device_ = config.mesh_device;
                device_local_size_ = config.buffer_size;
                device_local_layout_config_ = config.device_shard_layout;
            },
            [&](const ShardedBufferConfig& config) {
                const auto [mesh_height, mesh_width] = config.mesh_device->shape();
                const auto [global_buffer_height, global_buffer_width] = config.global_buffer_shape;
                const auto [shard_height, shard_width] = config.shard_shape;
                TT_FATAL(
                    (global_buffer_height % shard_height == 0) and (global_buffer_width % mesh_width == 0),
                    "The global buffer shape must be aligned to the shard shape:  global buffer shape: {}, {}, shard "
                    "shape: {}, {}",
                    global_buffer_height,
                    global_buffer_width,
                    shard_height,
                    shard_width);
                TT_FATAL(
                    shard_height * shard_width * config.mesh_device->num_devices() == global_buffer_height * global_buffer_width,
                    "Shards must be evenly distrbuted across all devices in the Mesh."
                );
                mesh_device_ = config.mesh_device;
                device_local_size_ = config.global_buffer_size / mesh_device_->num_devices();
                device_local_layout_config_ = config.device_shard_layout;
            }},
            mesh_buffer_config);

    if (address.has_value()) {
        address_ = address.value();
    } else {
        address_ = 0;
        auto buffer_to_allocate = this->get_shard_buffer(0, 0);
        address_ = tt::tt_metal::detail::AllocateBuffer(buffer_to_allocate.get());
        config_ = mesh_buffer_config;
    }
}

void MeshBuffer::deleter(MeshBuffer* mesh_buffer) {
    auto buffer_to_deallocate = mesh_buffer->get_shard_buffer(0, 0);
    tt::tt_metal::detail::DeallocateBuffer(buffer_to_deallocate.get());
}

std::shared_ptr<Buffer> MeshBuffer::get_shard_buffer(uint32_t logical_x, uint32_t logical_y) {
    DeviceAddr page_size = device_local_layout_config_.page_size;
    BufferType buffer_type = device_local_layout_config_.buffer_type;
    TensorMemoryLayout buffer_layout = device_local_layout_config_.buffer_layout;
    std::optional<ShardSpecBuffer> shard_parameters = device_local_layout_config_.shard_parameters;
    bool bottom_up = device_local_layout_config_.bottom_up;

    return Buffer::create(mesh_device_->get_device(logical_y, logical_x), address_, device_local_size_, page_size, buffer_type, buffer_layout, shard_parameters, bottom_up);
}

DeviceAddr MeshBuffer::global_size() const {
    return std::visit(  
        tt::stl::overloaded{
            [&](const ReplicatedBufferConfig& config) {
                return config.buffer_size;
            },
            [&](const ShardedBufferConfig& config) {
                return config.global_buffer_size;
            }
        },
        config_);
}

MeshBufferLayout MeshBuffer::global_layout() const {
    if (std::holds_alternative<ReplicatedBufferConfig>(config_)) {
        return MeshBufferLayout::REPLICATED;
    } else {
        return MeshBufferLayout::SHARDED;
    }
}

}  // namespace tt::tt_metal::distributed
