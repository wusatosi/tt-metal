// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    uint32_t result_addr = get_arg_val<uint32_t>(0);

    volatile tt_l1_ptr uint32_t* result_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_addr);

    WAYPOINT("WAIT");
    while (*result_addr_ptr != 39) {
        invalidate_l1_cache();
    }
    WAYPOINT("DONE");
}
