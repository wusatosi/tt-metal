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
    config_ = mesh_buffer_config;
    std::visit(  
        tt::stl::overloaded{
            [&](const ReplicatedBufferConfig& config) {
                mesh_device_ = config.mesh_device;
                device_local_size_ = config.buffer_size;
                device_local_layout_config_ = config.device_shard_layout;
            },
            [&](const ShardedBufferConfig& config) {
                const auto [global_buffer_width, global_buffer_height] = config.global_buffer_shape;
                const auto [shard_width, shard_height] = this->physical_shard_shape();
                // Account for replication here. Shard shape(dim) == buffer shape(dim) if replicated
                TT_FATAL(
                    (global_buffer_height % shard_height == 0) and (global_buffer_width % shard_width == 0),
                    "The global buffer shape must be aligned to the shard shape: global buffer shape: {}, {}, shard "
                    "shape: {}, {}",
                    global_buffer_height,
                    global_buffer_width,
                    shard_height,
                    shard_width);
                // Check needs to account for shard orientation. The scaling factor for replication depends on which
                // orientation we shard/replicate to when writing to device.
                auto num_shards = (global_buffer_width / shard_width) * (global_buffer_height / shard_height);
                if (std::get<0>(this->replicated_dims())) {
                    num_shards *= config.mesh_device->num_cols();
                }
                if (std::get<1>(this->replicated_dims())) {
                    num_shards *= config.mesh_device->num_rows();
                }
                // Assume row major shard orientation for now. Orientation determines the order in which data is written to Mesh.
                // We can always read from the sharded buffer in row major layout.
                TT_FATAL(num_shards <= config.mesh_device->num_devices(), "The sharded tensor does not fit on the Mesh.");
                mesh_device_ = config.mesh_device;
                device_local_size_ =  this->datum_size_bytes() * shard_height * shard_width;
                std::cout << "Shard Shape: " << shard_width << " " << shard_height << std::endl;
                std::cout << "Datum size bytes: " << this->datum_size_bytes() << std::endl; 
                device_local_layout_config_ = config.device_shard_layout;
            }},
            mesh_buffer_config);

    if (address.has_value()) {
        address_ = address.value();
    } else {
        address_ = 0;
        auto buffer_to_allocate = this->get_shard_buffer(0, 0);
        address_ = tt::tt_metal::detail::AllocateBuffer(buffer_to_allocate.get());
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

ShardSpecBuffer MeshBuffer::device_local_shard_spec() const {
    TT_FATAL(is_sharded(this->device_local_layout().buffer_layout), "Can only query the device local shard spec for a MeshBuffer sharded across cores");
    TT_FATAL(this->device_local_layout().shard_parameters.has_value(), "MeshBuffer is sharded across cores, but no shard parameters were specified.");
    return this->device_local_layout().shard_parameters.value();
}

ShardedBufferConfig MeshBuffer::global_shard_spec() const {
    TT_FATAL(this->global_layout() == MeshBufferLayout::SHARDED, "Can only query the global shard spec for a sharded MeshBuffer");
    return std::get<ShardedBufferConfig>(config_);
}

std::pair<size_t, size_t> MeshBuffer::physical_shard_shape() const {
    TT_FATAL(this->global_layout() == MeshBufferLayout::SHARDED, "Can only query physical shard shape for buffers sharded across the Mesh");
    auto sharded_config = std::get<ShardedBufferConfig>(config_);
    std::pair<size_t, size_t> physical_shard_shape = sharded_config.shard_shape;
    if (std::get<0>(physical_shard_shape) == 0) {
        std::get<0>(physical_shard_shape) = std::get<0>(sharded_config.global_buffer_shape);
    }
    if (std::get<1>(physical_shard_shape) == 0) {
        std::get<1>(physical_shard_shape) = std::get<1>(sharded_config.global_buffer_shape);
    }
    return physical_shard_shape;
}

std::pair<bool, bool> MeshBuffer::replicated_dims() const {
    TT_FATAL(this->global_layout() == MeshBufferLayout::SHARDED, "Can only query replicated dims for buffers sharded across the Mesh");
    auto sharded_config = std::get<ShardedBufferConfig>(config_);
    return {std::get<0>(sharded_config.shard_shape) == 0, std::get<1>(sharded_config.shard_shape) == 0};
}

uint32_t MeshBuffer::datum_size_bytes() const {
    // Limitation for now.
    TT_FATAL(this->global_layout() == MeshBufferLayout::SHARDED, "Can only query datum size for buffers sharded across the Mesh");
    auto sharded_config = std::get<ShardedBufferConfig>(config_);
    auto volume = std::get<0>(this->global_shard_spec().global_buffer_shape) * std::get<1>(this->global_shard_spec().global_buffer_shape);
    return (this->global_size() / volume);
}
}  // namespace tt::tt_metal::distributed
