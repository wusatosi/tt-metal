// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <ostream>
#include "tt_metal/common/bfloat16.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/numpy/functions.hpp"
#include "tt_metal/common/logger.hpp"

#include "common_tensor_test_utils.hpp"
#include "gtest/gtest.h"
#include "host_api.hpp"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/async_runtime.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn_test_fixtures.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

const CoreCoord start_core{0, 0};
const CoreCoord grid_size{2, 2};
const uint32_t num_cores = 4;

struct CreateShardedTensorInputs {
    ttnn::SimpleShape shape;
    DataType data_type;
    PageConfig page_config;
    MemoryConfig memory_config;
    std::vector<bfloat16> data;
};

struct CreateShardedTensorParams {
    CreateShardedTensorInputs inputs;
};
}  // namespace

class CreateShardedTensorTests : public ttnn::TTNNFixtureWithDevice,
                                 public ::testing::WithParamInterface<CreateShardedTensorParams> {};

TEST_P(CreateShardedTensorTests, ShardTensor) {
    int device_id = 0;
    Device* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    const auto& params = GetParam();
    const auto& input_shape = params.inputs.shape;
    const DataType dtype = params.inputs.data_type;
    const PageConfig page_config = params.inputs.page_config;
    MemoryConfig mem_cfg = params.inputs.memory_config;

    TensorLayout layout(params.inputs.data_type, params.inputs.page_config, params.inputs.memory_config);

    const uint32_t io_cq = 0;
    const auto input_buf_size_bytes = layout.compute_packed_buffer_size_bytes(input_shape);
    const auto host_buffer_datum_size_bytes = sizeof(uint32_t);
    const auto input_buf_size = input_buf_size_bytes / host_buffer_datum_size_bytes;

    auto host_data = std::make_shared<uint32_t[]>(input_buf_size);
    auto readback_data = std::make_shared<uint32_t[]>(input_buf_size);

    const auto max_value = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
    for (int i = 0; i < input_buf_size; i++) {
        host_data[i] = i % max_value;
    }

    auto tensor = tt::tt_metal::create_device_tensor(TensorSpec(input_shape, layout), device);
    ttnn::queue_synchronize(device->command_queue(io_cq));

    ttnn::write_buffer(io_cq, tensor, {host_data});
    ttnn::queue_synchronize(device->command_queue(io_cq));

    ttnn::read_buffer(io_cq, tensor, {readback_data});
    ttnn::queue_synchronize(device->command_queue(io_cq));

    for (int i = 0; i < input_buf_size; i++) {
        EXPECT_EQ(host_data[i], readback_data[i])
            << " Host data is: " << host_data[i] << ", Readback data is: " << readback_data[i]
            << ", On position: " << i;
    }

    tensor.deallocate();
}

INSTANTIATE_TEST_SUITE_P(
    TensorShardTests,
    CreateShardedTensorTests,
    ::testing::Values(
        // EXAMPLE 1: PASS
        //  This example should pass, since the second innermost dimension is divisible by 32
        CreateShardedTensorParams{CreateShardedTensorInputs{
            .shape = ttnn::SimpleShape{1, 2, 32, 64},
            .data_type = DataType::BFLOAT16,
            .page_config = PageConfig(Layout::TILE),
            .memory_config =
                MemoryConfig{
                    .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
                    .buffer_type = BufferType::L1,
                    .shard_spec =
                        ShardSpec{
                            num_cores_to_corerangeset(start_core, num_cores, grid_size, /*row_wise=*/true),
                            {32, 32},
                            ShardOrientation::ROW_MAJOR,
                            false,
                            ShardMode::PHYSICAL}}}},
        // EXAMPLE 2: FAIL
        //  This example shouldn't pass, since the second innermost dimension is not divisible by 32
        CreateShardedTensorParams{CreateShardedTensorInputs{
            .shape = ttnn::SimpleShape{2, 2, 16, 64},
            .data_type = DataType::BFLOAT16,
            .page_config = PageConfig(Layout::TILE),
            .memory_config =
                MemoryConfig{
                    .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
                    .buffer_type = BufferType::L1,
                    .shard_spec =
                        ShardSpec{
                            num_cores_to_corerangeset(start_core, num_cores, grid_size, /*row_wise=*/true),
                            {32, 32},
                            ShardOrientation::ROW_MAJOR,
                            false,
                            ShardMode::PHYSICAL}}}},
        // EXAMPLE 3 PASS
        // This example should pass, since the second innermost dimension is divisible by 32
        CreateShardedTensorParams{CreateShardedTensorInputs{
            .shape = ttnn::SimpleShape{2, 2, 32, 32},
            .data_type = DataType::BFLOAT16,
            .page_config = PageConfig(Layout::TILE),
            .memory_config =
                MemoryConfig{
                    .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
                    .buffer_type = BufferType::L1,
                    .shard_spec =
                        ShardSpec{
                            num_cores_to_corerangeset(start_core, num_cores, grid_size, /*row_wise=*/true),
                            {32, 32},
                            ShardOrientation::ROW_MAJOR,
                            false,
                            ShardMode::PHYSICAL}}}}  //,
        // EXAMPLE 4 FAIL
        // This example shouldn't pass, since the second innermost dimension is not divisible by 32
        // CreateShardedTensorParams{
        //    CreateShardedTensorInputs{
        //        .shape = ttnn::SimpleShape{2, 4, 16, 32},
        //        .data_type = DataType::BFLOAT16,
        //        .page_config = PageConfig(Layout::TILE),
        //        .memory_config =
        //            MemoryConfig{
        //                .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        //                .buffer_type = BufferType::L1,
        //                .shard_spec = ShardSpec{
        //                    num_cores_to_corerangeset(start_core, num_cores, grid_size, /*row_wise=*/true),
        //                    {32, 32},
        //                    ShardOrientation::ROW_MAJOR,
        //                    false,
        //                    ShardMode::PHYSICAL}
        //            }
        //    }
        //}
        ));
