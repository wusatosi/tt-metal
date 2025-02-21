// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <c_tensix_core.h>
#include "debug/dprint.h"
#include "debug/waypoint.h"

constexpr uint32_t atomic_start = get_compile_time_arg_val(14);

void kernel_main() {
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(atomic_start);
    noc_semaphore_set(sem, 0);
    DPRINT << "set sem " << (uint32_t)(sem) << ENDL();
    while (*sem == 0) {
        uint64_t dst_noc_multicast_addr = get_noc_multicast_addr(1, 2, 7, 11, atomic_start + 512);
        noc_async_write_multicast(atomic_start, dst_noc_multicast_addr, 16, 7 * 10, false);
        invalidate_l1_cache();
    }
    DPRINT << "done" << ENDL();
}
