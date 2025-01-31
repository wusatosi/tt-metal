// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    for (int i = 0; i < LOOP_COUNT; i++) {
        // Max unroll size
        DeviceZoneScopedN("TEST-FULL");
        {
            DeviceZoneScopedN("PRINT");
            DPRINT << p_reg[kernel_profiler::WALL_CLOCK_LOW_INDEX] << ENDL();
        }
        {
            DeviceZoneScopedN("NO-PRINT");
        }
    }
}
