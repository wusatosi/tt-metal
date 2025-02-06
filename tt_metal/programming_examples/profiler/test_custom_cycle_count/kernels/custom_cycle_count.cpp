// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

/**
 * LOOP_COUNT and LOOP_SIZE provide the ability to decide how many cycles this kernel takes.
 * With a large enough LOOP_COUNT and a LOOP_SIZEs within icache size, cycle count will be
 * very close to LOOP_COUNT x (LOOP_SIZE + loop_overhead). loop_overhead is 2 cycle 1 for
 * addi and 1 for branch if not zero.
 *
 * Keeping LOOP_SIZE constant and suitable for all 5 risc ichahes, The diff between to runs
 * with LOOP_COUNT and LOOP_COUNT + 1 should be the same across all riscs and it should be
 * LOOP_COUNT + 2 cycles
 *
 * More info on tt-metal issue #515
 *
 * https://github.com/tenstorrent/tt-metal/issues/515#issuecomment-1548434301
 */

volatile tt_reg_ptr uint32_t* p_reg_2 = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
void kernel_main() {
    {
        DeviceZoneScopedN("l1_chache") invalidate_l1_cache();
    }
    for (int i = 0; i < 1000000; i++) {
// Max unroll size
#pragma GCC unroll 65534
        for (int j = 0; j < LOOP_SIZE; j++) {
            uint32_t test2 = p_reg_2[0];
        }
    }
}
