// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>
#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
#ifndef OUT_SHARDED
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t output_tile_start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t stride = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles_w = get_compile_time_arg_val(3);

    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    const auto page_size = tile_bytes;
    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t l1_read_addr = get_read_ptr(cb_id_out);
    uint32_t current_tile = output_tile_start_id;

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_id_out, 1);
        noc_async_write_tile(current_tile, s, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, 1);
        current_tile++;
        if (current_tile % num_tiles_w == 0) {
            current_tile += stride;
        }
    }
#endif
}
