// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
/*
void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_blocks = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    DPRINT << "dst addr : " << dst_addr <<ENDL();
    DPRINT << "num_blocks : " << num_blocks <<ENDL();
    DPRINT << "start id: " << start_id <<ENDL();

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    const uint32_t num_tiles_per_2d = get_compile_time_arg_val(2);
    const uint32_t third_dim = get_compile_time_arg_val(3);
    const uint32_t number_blocks_per_core = get_compile_time_arg_val(4);

    DPRINT << "num_tiles_per_2d: " << num_tiles_per_2d <<ENDL();

#ifdef OUT_SHARDED
    cb_wait_front(cb_id_out, onetile);
#else

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

#ifdef BACKWARDS
    uint32_t end_id = - start_id - num_blocks;
    for (uint32_t dim = 0; dim > -third_dim; dim--) {
            for (uint32_t i = num_tiles_per_2d * dim - start_id;
                 i > end_id + num_tiles_per_2d * dim;
                 i--) {
#else
    uint32_t end_id = start_id + num_blocks;
    for (uint32_t dim = 0; dim < third_dim; dim++) {
            DPRINT << "dim is: " << dim <<ENDL();
            for (uint32_t i = num_tiles_per_2d * dim + start_id;
                 i < end_id + num_tiles_per_2d * dim;
                 i++) {
#endif
                cb_wait_front(cb_id_out, onetile);
                uint32_t l1_read_addr = get_read_ptr(cb_id_out);
                DPRINT << "WRITE I = :" << i <<ENDL();
                noc_async_write_tile(i, s, l1_read_addr);

                //auto* ptr_orig = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_read_addr);
                //for (uint32_t ii1 = 0; ii1 < 1024; ii1 = ii1+1) {
                //    if (ii1 % 16 == 0) {
                //    DPRINT << "CHECK HERE ";
                //    }
                //    DPRINT << "value at i1 = " << (uint32_t)ii1 <<  " is: " << BF16((uint16_t)ptr_orig[ii1]) <<
ENDL();
                //}

                noc_async_write_barrier();
                cb_pop_front(cb_id_out, onetile);
            }
        }
#endif
}
*/

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);
    uint32_t single_block_size_row_arg = get_arg_val<uint32_t>(2);
    uint32_t single_block_size_col_arg = get_arg_val<uint32_t>(3);

    // DPRINT << "dst addr : " << dst_addr <<ENDL();
    // DPRINT << "start id: " << start_id <<ENDL();
    // DPRINT << "single_block_size_row_arg: " << single_block_size_row_arg <<ENDL();
    // DPRINT << "single_block_size_col_arg " << single_block_size_col_arg <<ENDL();

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    const uint32_t num_tiles_per_2d = get_compile_time_arg_val(2);
    const uint32_t third_dim = get_compile_time_arg_val(3);
    uint32_t total_tiles_per_row = get_compile_time_arg_val(4);
    // DPRINT << "total_tiles_per_row : " << total_tiles_per_row <<ENDL();

    // DPRINT << "num_tiles_per_2d: " << num_tiles_per_2d <<ENDL();

#ifdef OUT_SHARDED
    cb_wait_front(cb_id_out, onetile);
#else

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

#ifdef BACKWARDS
    for (uint32_t dim = 0; dim > -third_dim; dim--) {
        for (uint32_t c = 0; c > -single_block_size_col_arg; c--) {
            for (uint32_t r = 0; r > -single_block_size_row_arg; r--) {
                uint32_t tile = -start_id + dim * num_tiles_per_2d + c * total_tiles_per_row + r;
#else
    for (uint32_t dim = 0; dim < third_dim; dim++) {
        // DPRINT << "dim : " << dim << ENDL();
        for (uint32_t c = 0; c < single_block_size_col_arg; c++) {
            for (uint32_t r = 0; r < single_block_size_row_arg; r++) {
                uint32_t tile = start_id + dim * num_tiles_per_2d + c * total_tiles_per_row + r;
                // DPRINT << "WRITING tile " << tile << ENDL();
#endif
                cb_wait_front(cb_id_out, onetile);
                uint32_t l1_read_addr = get_read_ptr(cb_id_out);

                noc_async_write_tile(tile, s, l1_read_addr);

                // auto* ptr_orig = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_read_addr);
                // for (uint32_t ii1 = 0; ii1 < 2048; ii1 = ii1+1) {
                //     if (ii1 % 16 == 0) {
                //     DPRINT << "CHECK HERE ";
                //     }
                //     DPRINT << "value at i1 = " << (uint32_t)ii1 <<  " is: " << BF16((uint16_t)ptr_orig[ii1]) <<
                //     ENDL();
                // }

                noc_async_write_barrier();
                cb_pop_front(cb_id_out, onetile);
            }
        }
    }
#endif
}
