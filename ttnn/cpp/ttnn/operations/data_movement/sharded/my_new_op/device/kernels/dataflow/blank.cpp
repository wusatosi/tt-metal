// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "debug/dprint.h"

void kernel_main() {
    uint32_t src0_cb_index = get_compile_time_arg_val(0);
    uint32_t src0_is_dram = get_compile_time_arg_val(1);
    uint32_t src1_cb_index = get_compile_time_arg_val(2);
    uint32_t src1_is_dram = get_compile_time_arg_val(3);
    uint32_t num_tiles = get_compile_time_arg_val(4);

    DPRINT << "src0_cb_index: " << src0_cb_index << ENDL();
    DPRINT << "src0_is_dram: " << src0_is_dram << ENDL();

    DPRINT << "src1_cb_index: " << src1_cb_index << ENDL();
    DPRINT << "src1_is_dram: " << src1_is_dram << ENDL();
    DPRINT << "num_tiles: " << num_tiles << ENDL();
}
