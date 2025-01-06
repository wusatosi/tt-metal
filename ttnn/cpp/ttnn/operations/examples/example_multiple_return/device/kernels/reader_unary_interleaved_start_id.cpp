// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "dprint.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker

    union {
        float f;
        uint32_t u;
    } a, b;
    a.f = -3.640625;
    b.f = 0.30078125;

    cb_reserve_back(cb_id_in0, 1);
    uint16_t* ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_id_in0));
    ptr[0] = static_cast<uint16_t>(a.u >> 16);
    cb_push_back(cb_id_in0, 1);

    cb_reserve_back(cb_id_in1, 1);
    ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_id_in1));
    ptr[0] = static_cast<uint16_t>(b.u >> 16);
    cb_push_back(cb_id_in1, 1);
}
