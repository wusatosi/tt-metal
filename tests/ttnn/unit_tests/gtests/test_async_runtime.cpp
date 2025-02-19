// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn_multi_command_queue_fixture.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"
#include <tt-metalium/bfloat16.hpp>
#include "ttnn/async_runtime.hpp"
#include <tt-metalium/event.hpp>
#include <cmath>

namespace tt::tt_metal {
namespace {

using MultiCommandQueueSingleDeviceFixture = ::ttnn::MultiCommandQueueSingleDeviceFixture;

TEST_F(MultiCommandQueueSingleDeviceFixture, TestAsyncRuntimeBufferDestructor) {
    // Test functionality for the buffer destructor, which will call deallocate asynchronously
    // We must ensure that the deallocate step, which can run after the buffer has been destroyed
    // does not rely on stale buffer state, after the buffer has been destroyed on host
    device_->enable_async(true);
    MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};

    uint32_t buf_size_datums = 1024 * 1024;
    uint32_t datum_size_bytes = 2;
    ttnn::Shape shape{1, 1, 1024, 1024};
    // Inside the loop, initialize a buffer with limited lifetime.
    // This will asynchronously allocate the buffer, wait for the allocation to complete (address to be assigned to the
    // buffer), destroy the buffer (which will asynchronously deallocate the buffer) in a loop
    TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg);
    TensorSpec tensor_spec(shape, tensor_layout);
    for (int loop = 0; loop < 100000; loop++) {
        auto input_buffer_dummy = tt::tt_metal::tensor_impl::allocate_buffer_on_device(device_, tensor_spec);
        device_->synchronize();
    }
}
}  // namespace
}  // namespace tt::tt_metal
