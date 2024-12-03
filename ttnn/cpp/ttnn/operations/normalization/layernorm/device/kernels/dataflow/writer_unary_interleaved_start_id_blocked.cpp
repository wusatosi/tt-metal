// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t tile_offset = get_arg_val<uint32_t>(2);

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t blk = get_compile_time_arg_val(1); // needed for correctness of softmax/LN kernels


    constexpr uint32_t cb_id_out0 = 16;
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out0);
    const DataFormat data_format = get_dataformat(cb_id_out0);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };
    // DPRINT << "WRITER: num_tiles = " << (uint16_t)num_tiles << ENDL();
    uint32_t tile_id = tile_offset;
    // for (uint32_t i = 0; i<num_tiles; i += blk) {
        // DPRINT << "WRITER: waiting for this many tiles: "<< (uint16_t)blk << ENDL();
        cb_wait_front(cb_id_out0, num_tiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        for (uint32_t j = 0; j<num_tiles; j++) {
            noc_async_write_tile(tile_id, s, l1_read_addr);
            tile_id++;
            l1_read_addr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, num_tiles);

    // }
}
