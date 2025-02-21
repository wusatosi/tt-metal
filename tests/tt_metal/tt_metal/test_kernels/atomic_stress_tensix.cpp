// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <c_tensix_core.h>
#include "debug/dprint.h"
#include "debug/waypoint.h"

constexpr uint64_t duration = (uint64_t)get_compile_time_arg_val(5) * 1000 * 1000 * 1000;
constexpr uint32_t atomic_start = get_compile_time_arg_val(14);

void kernel_main() {
    uint64_t done_time = c_tensix_core::read_wall_clock() + duration;
    uint64_t stall_time = 0;
    DPRINT << "starting " << duration << ENDL();
    while (c_tensix_core::read_wall_clock() < done_time) {
        invalidate_l1_cache();
        uint64_t noc_write_addr = NOC_XY_ADDR(NOC_X(3), NOC_Y(1), atomic_start + 4);
        noc_semaphore_inc(noc_write_addr, 1);
        noc_write_addr = NOC_XY_ADDR(NOC_X(3), NOC_Y(1), atomic_start + 16);
        noc_async_write_one_packet(atomic_start, noc_write_addr, 4);
        (*(uint32_t*)atomic_start)++;
    }
    DPRINT << "done" << ENDL();
}
