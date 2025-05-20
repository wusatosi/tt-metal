// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_fixture.hpp"

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <dev_mem_map.h>

#include <string>

#include <gtest/gtest.h>

namespace {
void RunOneTest(WatcherFixture* fixture, IDevice* device, unsigned usage, bool warning) {
    static const struct {
        const char *name;
        unsigned space;
    } tags[] = {
        {"brisc", MEM_BRISC_STACK_SIZE},
        {"ncrisc", MEM_NCRISC_STACK_SIZE},
        {"trisc0", MEM_TRISC0_STACK_SIZE},
        {"trisc1", MEM_TRISC1_STACK_SIZE},
        {"trisc2", MEM_TRISC2_STACK_SIZE},
    };
    const std::string path = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_stack.cpp";
    auto msg = [&](std::vector<std::string> &usages,
                   const char *cpu, unsigned space, unsigned usage, bool warning) {
        if (!warning)
            return;
        bool overflow = !usage;
        char fraction[64];
        snprintf (fraction, sizeof(fraction), "%4u/%4u", space - usage, space);
        std::string msg;

        if (usages.empty())
            usages.push_back("Stack usage summary:");
        msg.clear();
        msg.append(cpu).append(" highest stack usage: ").append(fraction).append(", on core");
        usages.push_back(msg);
        msg.clear();
        msg.append("running kernel ").append(path).append(overflow ? " (OVERFLOW)" : " (Close to overflow)");
        usages.push_back(msg);
    };
    
    // Set up program
    Program program = Program();
    CoreCoord coord = {0, 0};
    std::vector<uint32_t> compile_args{usage};
    vector<string> expected, overflows;

    // Run a kernel that posts waypoints and waits on certain gating values to be written before
    // posting the next waypoint.
    auto brisc_kid = CreateKernel(
        program,
        path,
        coord,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default,
            .compile_args = compile_args});
    msg(expected, tags[0].name, tags[0].space, usage, warning);

    auto ncrisc_kid = CreateKernel(
        program,
        path,
        coord,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
            .compile_args = compile_args});
    msg(expected, tags[1].name, tags[1].space, usage, warning);

    auto trisc_kid = CreateKernel(program, path, coord, ComputeConfig{.compile_args = compile_args});
    for (unsigned ix = 0; ix != 2; ix++) {
        msg(expected, tags[2 + ix].name, tags[2 + ix].space, usage, warning);
    }

    // Also run on ethernet cores if they're present
    bool has_eth_cores = false && !device->get_active_ethernet_cores(true).empty();
    bool has_idle_eth_cores = false && fixture->IsSlowDispatch() &&
        !device->get_inactive_ethernet_cores().empty();
    // FIXME: Implement eth

    fixture->RunProgram(device, program);
    sleep(1);

    EXPECT_TRUE(
        FileContainsAllStringsInOrder(
            fixture->log_file_name,
            expected
            )
        );
}

template<uint32_t Usage, bool Msg>
void RunTest(WatcherFixture* fixture, IDevice* device) {
    RunOneTest(fixture, device, Usage, Msg);
}

} // namespace

TEST_F(WatcherFixture, TestWatcherStackUsage0) {
    for (IDevice* device : this->devices_) {
        this->RunTestOnDevice(RunTest<0, true>, device);
    }
}

TEST_F(WatcherFixture, TestWatcherStackUsage16) {
    for (IDevice* device : this->devices_) {
        this->RunTestOnDevice(RunTest<16, true>, device);
    }
}

TEST_F(WatcherFixture, TestWatcherStackUsage128) {
    for (IDevice* device : this->devices_) {
        this->RunTestOnDevice(RunTest<128, false>, device);
    }
}
