// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/llrt/tt_memory.h"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/detail/kernel_cache.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/device/device_pool.hpp"
#include "llrt/hal.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;

std::string get_latest_kernel_binary_path(const string& kernel_root_path, const std::shared_ptr<Kernel>& kernel) {
    TT_FATAL(kernel != nullptr, "Error");
    TT_FATAL(std::filesystem::exists(kernel_root_path + kernel->name()), "Error");

    std::filesystem::path kernel_path{kernel_root_path + kernel->name()};
    std::filesystem::file_time_type ftime = std::filesystem::last_write_time(*kernel_path.begin());
    std::string latest_hash;
    for (auto const& dir_entry : std::filesystem::directory_iterator{kernel_path}) {
        auto kbtime = std::filesystem::last_write_time(dir_entry.path());
        if (kbtime > ftime) {
            ftime = kbtime;
            latest_hash = dir_entry.path().filename().string();
        }
    }
    TT_FATAL(not latest_hash.empty(), "Error");
    return kernel->name() + "/" + latest_hash;
}

void construct_program(Program& program, Device* device, CoreCoord& core) {
    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2048;
    uint32_t dram_buffer_size =
        single_tile_size * num_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    tt_metal::InterleavedBufferConfig buff_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    auto src_dram_buffer = CreateBuffer(buff_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();
    auto dst_dram_buffer = CreateBuffer(buff_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();


    // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the
    // input CB CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to
    // math kernel, input CB and reader
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 8;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 1;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    auto unary_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {
        uint(num_tiles)  // per_core_tile_cnt
    };

    auto eltwise_unary_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args});
}

int main(int argc, char** argv) {
    bool pass = true;
    return 0;
}
