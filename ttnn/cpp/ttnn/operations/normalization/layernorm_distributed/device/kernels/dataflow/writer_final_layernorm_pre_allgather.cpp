// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel writes the final merged layernorm statistics to DRAM.
 * It receives data from the merge core and writes it to the final output.
 */

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    DPRINT << "WF_KRNL_START" << ENDL();
    // Get runtime arguments
    const uint32_t dest_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    // const uint32_t tiles_per_batch = get_arg_val<uint32_t>(2); // Unused?
    // const uint32_t semaphore_id = get_arg_val<uint32_t>(3);    // Unused?

    // DPRINT << "WF_START dest=" << dest_addr << " num_tiles=" << num_tiles << ENDL();

    constexpr uint32_t cb_out = tt::CBIndex::c_14;  // Output circular buffer
    constexpr bool dst_is_dram = get_compile_time_arg_val(0);
    constexpr bool is_rmsnorm = get_compile_time_arg_val(1);

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const DataFormat data_format = get_dataformat(cb_out);

    const InterleavedAddrGenFast<dst_is_dram> dst_accessor = {
        .bank_base_address = dest_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t tile_cols_per_sequence = is_rmsnorm ? 1 : 2;  // 1 for RMSNorm, 2 for LayerNorm
    uint32_t tile_id = 0;

    // DPRINT << "WF_INIT_DONE" << ENDL();

    for (uint32_t i = 0; i < num_tiles; i += tile_cols_per_sequence) {
        DPRINT << "WF_WAIT_COMPUTE tiles=" << tile_cols_per_sequence << ENDL();
        cb_wait_front(cb_out, tile_cols_per_sequence);
        // DPRINT << "WF_INPUT_RECEIVED" << ENDL();

        uint32_t l1_read_addr = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < tile_cols_per_sequence; j++) {
            // DPRINT << "WF_WRITE_TILE" << tile_id << ENDL();
            noc_async_write_tile(tile_id, dst_accessor, l1_read_addr);
            tile_id++;
            l1_read_addr += tile_bytes;
        }

        // DPRINT << "WF_WAIT_WRITE" << ENDL();
        noc_async_write_barrier();

        // DPRINT << "WF_POP_TILES" << tile_cols_per_sequence << ENDL();
        cb_pop_front(cb_out, tile_cols_per_sequence);
    }

    DPRINT << "WF_KRNL_END" << ENDL();
}
