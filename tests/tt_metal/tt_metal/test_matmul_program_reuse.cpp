// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat8.hpp"
#include "tt_metal/detail/util.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;

int main(int argc, char** argv) {
    bool pass = true;

    std::map<chip_id_t, tt_metal::Device*> devices_ = tt::tt_metal::detail::CreateDevices({0, 4, 1, 5, 2, 6, 3, 7});

    CoreCoord worker_grid_size = devices_[0]->compute_with_storage_grid_size();
    std::vector<std::shared_ptr<Buffer>> input_buffers = {};
    std::vector<std::shared_ptr<Buffer>> output_buffers = {};
    uint32_t single_tile_size = tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

    uint32_t num_tiles = 1;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;  // num_tiles of BFP8_B
    for (auto device : devices_) {
        auto dev = device.second;
        tt_metal::InterleavedBufferConfig dram_config{
            .device = dev,
            .size = dram_buffer_size,
            .page_size = dram_buffer_size,
            .buffer_type = tt_metal::BufferType::DRAM};

        for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
            input_buffers.push_back(CreateBuffer(dram_config));
            output_buffers.push_back(CreateBuffer(dram_config));
        }
    }

    tt_metal::Program program = tt_metal::CreateProgram();
    auto first_col = CoreRange({0, 0}, {0, worker_grid_size.y - 1});
    auto reader_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/reader_writer_in_risc.cpp",
        first_col,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto sem_scaling_factor = 2;
    tt_metal::CreateSemaphore(program, first_col, sem_scaling_factor);

    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(dram_buffer_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);

    for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
        CoreCoord curr_core = {0, row_idx};
        tt_metal::SetRuntimeArgs(
            program,
            reader_writer_kernel,
            curr_core,
            {input_buffers.at(row_idx)->address(),
             output_buffers.at(row_idx)->address(),
             0, /* src_bank_id */
             0, /* dst_bank_id */
             64,
             32,
             32,
             0,
             16});
        CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, curr_core, cb_src0_config);
    }

    std::vector<uint32_t> src_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 1);
    std::size_t buffer_idx = 0;
    for (auto device : devices_) {
        auto dev = device.second;
        for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
            EnqueueWriteBuffer(dev->command_queue(), input_buffers.at(buffer_idx), src_vec, false);
            buffer_idx++;
        }
    }
    for (int iter = 0; iter < 2; iter++) {
        if (iter) {
            auto& rtas = GetRuntimeArgs(program, reader_writer_kernel);
            for (auto core : first_col) {
                rtas[core.x][core.y].at(4) = 2 * rtas[core.x][core.y].at(4);
            }
        }
        for (auto device : devices_) {
            auto dev = device.second;
            EnqueueProgram(dev->command_queue(), program, false);
        }

        buffer_idx = 0;
        for (auto device : devices_) {
            auto dev = device.second;
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                std::vector<bfloat16> dst_vec = {};
                EnqueueReadBuffer(dev->command_queue(), output_buffers.at(buffer_idx), dst_vec, true);
                buffer_idx++;

                for (auto i : dst_vec) {
                    // TT_ASSERT(i.to_float() )
                    std::cout << i.to_float() << " ";
                }
            }
        }
    }
}
