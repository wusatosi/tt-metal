// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>
#include <utility>

#include "device_fixture.hpp"
#include "multi_device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using std::vector;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;



TEST_F(DeviceSingleCardFastSlowDispatchFixture, TestSimpleNoCAsyncReadProgram) {
    uint32_t NUM_PROGRAMS = 100;
    uint32_t MAX_LOOP = 10000;
    uint32_t page_size = 1024;

    // Make random
    auto random_seed = 0; // (unsigned int)time(NULL);
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    log_info(tt::LogTest, "Using Test Seed: {}", seed);
    srand(seed);

    CoreCoord worker_grid_size = this->device_->compute_with_storage_grid_size();
    CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet cr_set(cr);

    std::map<string, string> defines;
    defines["READ"] = "1";
    defines["WRITE"] = "1";
    defines["ATOMIC"] = "1";
    defines["POSTED_WRITE"] = "1";

    log_info(tt::LogTest, "Starting compile of {} programs now.", NUM_PROGRAMS);

    vector<Program> programs;
    for (uint32_t i = 0; i < NUM_PROGRAMS; i++) {
        programs.push_back(Program());
        Program& program = programs.back();

        // brisc
        uint32_t BRISC_OUTER_LOOP, BRISC_MIDDLE_LOOP, BRISC_INNER_LOOP, NUM_CBS, NUM_SEMS;
        bool USE_MAX_RT_ARGS;

        if (i % 10 == 0) {
            log_info(tt::LogTest, "Compiling program {} of {}", i + 1, NUM_PROGRAMS);
        }

        CircularBufferConfig cb_config = CircularBufferConfig(page_size, {{0, tt::DataFormat::Float16_b}}).set_page_size(0, page_size);
        auto cb = CreateCircularBuffer(program, cr_set, cb_config);

        vector<uint32_t> compile_args = {MAX_LOOP, page_size};

        auto brisc_kernel = CreateKernel(
            program, "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_reader_writer.cpp", cr_set, DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = compile_args, .defines = defines});

        auto ncrisc_kernel = CreateKernel(
            program, "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_reader_writer.cpp", cr_set, DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = compile_args, .defines = defines});

        for(int core_idx_y = 0; core_idx_y < worker_grid_size.y; core_idx_y++) {
            for(int core_idx_x = 0; core_idx_x < worker_grid_size.x; core_idx_x++) {
                CoreCoord core = {(std::size_t) core_idx_x, (std::size_t) core_idx_y};
                CoreCoord neighbour_core = {core_idx_x == worker_grid_size.x - 1 ? 0 : core_idx_x + 1, core_idx_y};
                CoreCoord neighbour_core_physical = device_->worker_core_from_logical_core(neighbour_core);
                std::vector<uint32_t> rt_args = {
                    (std::uint32_t) neighbour_core_physical.x,
                    (std::uint32_t) neighbour_core_physical.y,
                };
                tt::tt_metal::SetRuntimeArgs(program, brisc_kernel, core, rt_args);
                tt::tt_metal::SetRuntimeArgs(program, ncrisc_kernel, core, rt_args);
            }
        }

        tt::tt_metal::detail::CompileProgram(this->device_, program);
    }

    log_info(tt::LogTest, "Running {} programs for cache warmup.", programs.size());
    // This loop caches program and runs
    for (Program& program: programs) {
        if (this->slow_dispatch_) {
            tt::tt_metal::detail::LaunchProgram(this->device_, program);
        } else {
            EnqueueProgram(this->device_->command_queue(), program, false);
        }
    }
    if (!this->slow_dispatch_) {
        Finish(this->device_->command_queue());
        log_info(tt::LogTest, "Finish FD runs");
    } else {
        log_info(tt::LogTest, "Finish SD runs");
    }
}
