// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/dispatch/command_queue_interface.hpp"
#include "tt_metal/impl/sub_device/sub_device_types.hpp"
#include "buffer.hpp"

namespace tt::tt_metal {

namespace buffer_utils {

struct BufferDispatchConstants {
    uint32_t issue_queue_cmd_limit;
    uint32_t max_prefetch_cmd_size;
    uint32_t max_data_sizeB;
};

struct InterleavedBufferDispatchParams {
    uint32_t total_pages_to_write;
    uint32_t write_partial_pages;
    uint32_t padded_page_size;
    uint32_t padded_buffer_size;
    uint32_t pages_to_write;
    uint32_t max_num_pages_to_write;
    std::array<uint32_t, dispatch_constants::DISPATCH_MESSAGE_ENTRIES> expected_num_workers_completed;
    uint32_t address;
    uint32_t dst_page_index;
    bool issue_wait;
    IDevice* device;
    uint32_t cq_id;
};

struct ShardedBufferDispatchParams {
    bool width_split;
    std::shared_ptr<const BufferPageMapping> buffer_page_mapping;
    uint32_t num_total_pages;
    uint32_t max_pages_per_shard;
    uint32_t padded_page_size;
    uint32_t pages_to_write;
    CoreCoord core;
    std::array<uint32_t, dispatch_constants::DISPATCH_MESSAGE_ENTRIES> expected_num_workers_completed;
    uint32_t address;
    uint32_t dst_page_index;
    bool issue_wait;
    IDevice* device;
    uint32_t cq_id;
};

BufferDispatchConstants generate_buffer_dispatch_constants(
    const SystemMemoryManager& sysmem_manager, CoreType dispatch_core_type, uint32_t cq_id);

ShardedBufferDispatchParams initialize_sharded_buf_dispatch_params(
    Buffer& buffer,
    uint32_t cq_id,
    std::array<uint32_t, dispatch_constants::DISPATCH_MESSAGE_ENTRIES> expected_num_workers_completed);

std::vector<CoreCoord> get_cores_for_sharded_buffer(const ShardedBufferDispatchParams& dispatch_params, Buffer& buffer);

void write_sharded_buffer_to_core(
    const void* src,
    uint32_t core_id,
    Buffer& buffer,
    ShardedBufferDispatchParams& dispatch_params,
    BufferDispatchConstants& buf_dispatch_constants,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    const std::vector<CoreCoord>& cores);

InterleavedBufferDispatchParams initialize_interleaved_buf_dispatch_params(
    Buffer& buffer,
    BufferDispatchConstants& buf_dispatch_constants,
    uint32_t cq_id,
    std::array<uint32_t, dispatch_constants::DISPATCH_MESSAGE_ENTRIES> expected_num_workers_completed);

void write_interleaved_buffer_to_device(
    const void* src,
    InterleavedBufferDispatchParams& dispatch_params,
    Buffer& buffer,
    BufferDispatchConstants& buf_dispatch_constants,
    tt::stl::Span<const SubDeviceId> sub_device_ids);

}  // namespace buffer_utils

}  // namespace tt::tt_metal
