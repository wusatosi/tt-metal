// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "core_coord.hpp"
#include "kernels/data_types.hpp"
#include "logger.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/hostdevcommon/profiler_common.h"

using std::vector;
using namespace tt;

int main(int argc, char** argv) {
    std::vector<std::string> input_args(argv, argv + argc);
    auto packet_size = test_args::get_command_option_uint32(input_args, "-p", 4096);
    auto packet_count = test_args::get_command_option_uint32(input_args, "-c", 4);
    auto iterations = test_args::get_command_option_uint32(input_args, "-i", 2);

    bool pass = true;
    try {
        int device_id = 0;
        tt_metal::Device* device = tt_metal::CreateDevice(device_id);
        CommandQueue& cq = device->command_queue();
        tt_metal::Program program = tt_metal::CreateProgram();

        //-- setup worker coordinates -----------------------------------------//
        std::vector<std::tuple<CoreCoord,CoreCoord,uint32_t,uint32_t, tt_metal::DataMovementProcessor, tt_metal::NOC>> transfers = {
            {{0,0}, {0,0},4096,4,tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::NOC_0},
            {{0,0}, {0,1},4096,4,tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::NOC_0},
            {{0,0}, {1,0},4096,4,tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::NOC_0},
            {{0,0}, {1,1},4096,4,tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::NOC_0},

            {{0,1}, {0,2},4096,4,tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::NOC_0},
            {{0,1}, {0,3},4096,4,tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::NOC_0},
            {{0,1}, {1,2},4096,4,tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::NOC_0},
            {{0,1}, {1,3},4096,4,tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::NOC_0},

            {{1,0}, {2,0},4096,4,tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::NOC_0},
            {{1,0}, {2,1},4096,4,tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::NOC_0},
            {{1,0}, {3,0},4096,4,tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::NOC_0},
            {{1,0}, {3,1},4096,4,tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::NOC_0},

            {{1,1}, {2,2},4096,4,tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::NOC_0},
            {{1,1}, {2,3},4096,4,tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::NOC_0},
            {{1,1}, {3,2},4096,4,tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::NOC_0},
            {{1,1}, {3,3},4096,4,tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::NOC_0},

            {{0,0}, {0,0},4096,4,tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::NOC_1},
            {{0,0}, {0,1},4096,4,tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::NOC_1},
            {{0,0}, {1,0},4096,4,tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::NOC_1},
            {{0,0}, {1,1},4096,4,tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::NOC_1},

            {{0,1}, {0,2},4096,4,tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::NOC_1},
            {{0,1}, {0,3},4096,4,tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::NOC_1},
            {{0,1}, {1,2},4096,4,tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::NOC_1},
            {{0,1}, {1,3},4096,4,tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::NOC_1},

            {{1,0}, {2,0},4096,4,tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::NOC_1},
            {{1,0}, {2,1},4096,4,tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::NOC_1},
            {{1,0}, {3,0},4096,4,tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::NOC_1},
            {{1,0}, {3,1},4096,4,tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::NOC_1},

            {{1,1}, {2,2},4096,4,tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::NOC_1},
            {{1,1}, {2,3},4096,4,tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::NOC_1},
            {{1,1}, {3,2},4096,4,tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::NOC_1},
            {{1,1}, {3,3},4096,4,tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::NOC_1},
        };

        //-- setup defines ---------------------------------------------------//
        const uint32_t ITERATIONS = iterations;
        const uint32_t PAGE_COUNT = packet_count;
        const uint32_t PAGE_SIZE = packet_size;

        //-- circular buffer config --------------------------------------------//
        constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
        const uint32_t total_bytes = 4096*64;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(total_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, packet_size);

        //-- program worker cores --------------------------------------------//
        auto kernel_cpp_file = "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/congested_read.cpp";

        std::set<CoreCoord> seen;
        for (auto [logical_src_loc,logical_dst_loc,packet_size,num_packets, proc_type, noc_type] : transfers) {

            auto phys_dst_loc = device->physical_core_from_logical_core(logical_dst_loc, CoreType::WORKER);
            auto phys_src_loc =
                device->physical_core_from_logical_core(logical_src_loc, CoreType::WORKER);
            std::vector<uint32_t> runtime_args = {phys_src_loc.x, phys_src_loc.y};

            log_info(
                "Setting up reader at {},{} ({}) reading from {},{} on NOC {}...  ps={} num={}",
                phys_dst_loc.x,
                phys_dst_loc.y,
                magic_enum::enum_name(proc_type),
                phys_src_loc.x,
                phys_src_loc.y,
                magic_enum::enum_name(noc_type),
                packet_size, num_packets);

            const std::map<string, string> common_defines = {
                {"ITERATIONS", std::to_string(1)},
                {"PAGE_COUNT", std::to_string(num_packets)},
                {"PAGE_SIZE", std::to_string(packet_size)}};

            auto kernel_handle = tt_metal::CreateKernel(
                program,
                kernel_cpp_file,
                logical_dst_loc,
                tt_metal::DataMovementConfig{.processor = proc_type, .noc = noc_type, .defines = common_defines});
            tt_metal::SetRuntimeArgs(program, kernel_handle, logical_dst_loc, runtime_args);

            if (!seen.count(logical_dst_loc)) {
                tt_metal::CreateCircularBuffer(program, logical_dst_loc, cb_src0_config);
                seen.insert(logical_dst_loc);
            }
            if (!seen.count(logical_src_loc)) {
                tt_metal::CreateCircularBuffer(program, logical_src_loc, cb_src0_config);
                seen.insert(logical_src_loc);
            }
        }

        //-- warmup ----------------------------------------------------------//
        EnqueueProgram(cq, program, false);
        Finish(cq);
        tt_metal::detail::DumpDeviceProfileResults(device);

        //-- run -------------------------------------------------------------//
        //EnqueueProgram(cq, program, false);
        //Finish(cq);
        //tt_metal::detail::DumpDeviceProfileResults(device);
        //auto start = std::chrono::system_clock::now();
        //for (int i = 0; i < 100'000; i++) {
        //    EnqueueProgram(cq, program, false);
        //}
        //Finish(cq);
        //auto end = std::chrono::system_clock::now();
        //float etime_ns = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        //const float dispatch_overhead = 10800;
        //log_info("            elapsed time: {:.1f} ns", etime_ns);
        //log_info(" elapsed time per iteration: {:.1f} ns", (etime_ns / 100'000.0) - dispatch_overhead);


        // log_info("   oneway latency: {:.1f} ns", etime_ns / (2 * ITERATIONS));
        //// measured by profiling loop with local semaphore increment
        // constexpr float loop_overhead_ns = 13.5;
        // log_info("--------------------------------------------", etime_ns / (2 * ITERATIONS));
        // log_info(
        //     "   oneway latency, _minus loop overhead_: {:.1f} ns", (etime_ns / (2 * ITERATIONS)) - loop_overhead_ns);

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception& e) {
        pass = false;
        log_fatal(e.what());
    }
}
