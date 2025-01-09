// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/buffers/buffer_constants.hpp"
#include "tt_metal/tt_stl/overloaded.hpp"
#include "tt_metal/distributed/mesh_device.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/buffers/buffer_dispatch_utils.hpp"

namespace tt::tt_metal::distributed {

enum class MeshBufferLayout : uint8_t { REPLICATED, SHARDED };

// Specifies how a buffer is laid out across Memory Banks within a single device.
struct DeviceLocalLayoutConfig {
    DeviceAddr page_size;

    // Can be DRAM, L1, SYSTEM_MEMORY, L1_SMALL, TRACE.
    BufferType buffer_type = BufferType::DRAM;

    // Can be INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED or BLOCK_SHARDED.
    TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED;

    // Must be set for sharded buffer layouts.
    std::optional<ShardSpecBuffer> shard_parameters = std::nullopt;

    // The direction in which memory for this buffer is allocated.
    bool bottom_up = false;
};

// Specifies MeshBuffer that is replicated across the virtual mesh.
struct ReplicatedBufferConfig {
    // The MeshDevice on which the associated buffer will be allocated
    MeshDevice* mesh_device;
    // Each device will get a buffer of this size.
    DeviceAddr buffer_size;
    // Specifies how each replicated shard is laid out across Memory Banks on a single device
    DeviceLocalLayoutConfig device_shard_layout;
};

// Specifies sharded MeshBuffer.
struct ShardedBufferConfig {
    // The MeshDevice on which the associated buffer will be allocated
    MeshDevice* mesh_device;
    // Global buffer size. Each device will get a fraction of this size.
    DeviceAddr global_buffer_size;
    // The shape of each shard sent to each device - used to determine the data-distribution scheme
    std::pair<size_t, size_t> shard_shape = {0, 0};
    // Global shape of the buffer; at metal-level, we expect the shape to be aligned with the mesh shape.
    std::pair<size_t, size_t> global_buffer_shape = {0, 0};
    // Specifies how each replicated shard is laid out across Memory Banks on a single device
    DeviceLocalLayoutConfig device_shard_layout;
};

using MeshBufferConfig = std::variant<ReplicatedBufferConfig, ShardedBufferConfig>;

class MeshBuffer {
public:
    static std::shared_ptr<MeshBuffer> create(const MeshBufferConfig& config, std::optional<DeviceAddr> address = std::nullopt);
    MeshDevice* mesh_device() const { return mesh_device_; }
    DeviceAddr device_local_size() const { return device_local_size_; }
    DeviceAddr global_size() const;
    DeviceAddr address() const { return address_; };
    MeshBufferLayout global_layout() const;
    DeviceLocalLayoutConfig device_local_layout() const { return device_local_layout_config_; }
    std::shared_ptr<Buffer> get_shard_buffer(uint32_t logical_x, uint32_t logical_y);
    ShardSpecBuffer device_local_shard_spec() const;
    ShardedBufferConfig global_shard_spec() const;
    void deallocate() {}

private:
    MeshBuffer(const MeshBufferConfig& config, std::optional<DeviceAddr> address = std::nullopt);
    static void deleter(MeshBuffer* mesh_buffer);
    DeviceAddr device_local_size_ = 0;
    MeshDevice* mesh_device_ = nullptr;
    DeviceAddr address_ = 0;
    DeviceLocalLayoutConfig device_local_layout_config_;
    MeshBufferConfig config_;
};

}  // namespace tt::tt_metal::distributed
