// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/device/device.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    // get program/device
    int device_id = 0;
    Device* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    // initialize source data - vector with shape (16, 1)
    constexpr uint32_t M = 16384;
    constexpr uint32_t N = 1;
    uint32_t num_values = M * N;
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    std::vector<bfloat16> src_vec(num_values, bfloat16(0.0f));
    // source vector - {2, 4, 6, ... , 28, 30, 32}
    for (uint32_t i = 0; i < src_vec.size(); i++) {
        src_vec[i] = bfloat16((i + 1) * 2);
    }
    constexpr size_t data_size = sizeof(bfloat16);

    // designate 4 cores for utilization - cores (0,0), (0,1), (0,2), (0,3)
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {7, 7};
    uint32_t num_cores = 64;
    CoreRange cores(start_core, end_core);

    // define shard specs - 16 values total and 4 cores -> 4 values per core
    uint32_t shard_height = num_values / num_cores / data_size;
    uint32_t shard_width = N;
    uint32_t shard_size = shard_height * shard_width;
    uint32_t input_unit_size = sizeof(uint32_t);
    uint32_t shard_width_bytes = shard_width * data_size;
    uint32_t num_units_per_row = shard_width * input_unit_size;
    uint32_t padded_offset_bytes = align(input_unit_size, device->get_allocator_alignment());

    // configure and create interleaved DRAM buffer to insert source data into
    uint32_t src_buffer_size = input_unit_size * num_values / data_size;
    tt_metal::InterleavedBufferConfig input_dram_config{
        .device = device,
        .size = src_buffer_size,
        .page_size = input_unit_size,
        .buffer_type = tt_metal::BufferType::DRAM};
    std::shared_ptr<tt::tt_metal::Buffer> src_buffer = CreateBuffer(input_dram_config);
    uint32_t src_addr = src_buffer->address();

    tt_metal::InterleavedBufferConfig l1_config{
        .device = device, .size = 256 * 4096, .page_size = 4096, .buffer_type = tt_metal::BufferType::L1};

    std::shared_ptr<tt_metal::Buffer> l1_buffer = CreateBuffer(l1_config);

    // configure and create circular buffers with the same address on each of the designated cores
    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t input_cb_index = CBIndex::c_0;
    CircularBufferConfig input_cb_config =
        CircularBufferConfig(shard_size * input_unit_size, {{input_cb_index, cb_data_format}})
            .set_page_size(input_cb_index, input_unit_size);
    auto cb_input = tt_metal::CreateCircularBuffer(program, cores, input_cb_config);

    // create data movement kernel to shard data
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)input_cb_index, (std::uint32_t)src_is_dram};
    auto reader_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/sharding/kernels/reader_sharded_rm.cpp",
        cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});

    // set runtime arguments for each core
    uint32_t curr_idx_h = 0;
    uint32_t curr_idx_w = 0;
    CoreRange physical_cores{
        device->worker_core_from_logical_core(start_core), device->worker_core_from_logical_core(end_core)};
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i % 8, i / 8};

        bool is_multicast = ((core.x == 2) && (core.y == 6)) || ((core.x == 2) && (core.y == 7));
        tt_metal::SetRuntimeArgs(
            program,
            reader_id,
            core,
            {src_addr,
             input_unit_size,
             shard_height,
             shard_width_bytes,
             padded_offset_bytes,
             curr_idx_h,
             i,
             is_multicast,
             l1_buffer->address(),
             /* mcast address */ device->get_noc_multicast_encoding(NOC::RISCV_0_default, physical_cores),
             63});
        curr_idx_w += input_unit_size;
        if (curr_idx_w >= num_units_per_row) {
            curr_idx_w = 0;
            curr_idx_h += shard_height;
        }
    }
#if 0

    printf("Original tensor values: ");
    for (uint32_t src_vec_idx = 0; src_vec_idx < src_vec.size(); src_vec_idx++) {
        printf("%0.f ", src_vec[src_vec_idx].to_float());
    }
    printf("\n");
#endif

    // start/finish program and close device
    EnqueueWriteBuffer(cq, src_buffer, src_vec.data(), false);
    EnqueueProgram(cq, program, false);
    Finish(cq);
    CloseDevice(device);
}
