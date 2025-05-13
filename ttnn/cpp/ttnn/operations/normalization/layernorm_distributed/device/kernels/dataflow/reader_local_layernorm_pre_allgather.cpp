// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel reads the layernorm inputs from interleaved dram for the local computation stage.
 * It handles dimension sharding by reading only the subset of dimension tiles assigned to this core.
 */

#include <stdint.h>
#include "dataflow_api.h"
// #include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
// #include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "debug/dprint.h"
void kernel_main() {
    DPRINT << "RL_KRNL_START" << ENDL();
    // Args for Interleaved Reading
    const uint32_t src_addr = get_arg_val<uint32_t>(0);             // Base address of input tensor
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(1);        // Num rows (NC*Ht) this core handles
    const uint32_t Wt = get_arg_val<uint32_t>(2);                   // Full width in tiles
    const uint32_t Wt_offset = get_arg_val<uint32_t>(3);            // Tile offset in W dim for this core's shard
    const uint32_t tiles_per_dim_shard = get_arg_val<uint32_t>(4);  // Num tiles in W dim this core reads
    const uint32_t start_tile_id = get_arg_val<uint32_t>(5);  // Starting tile ID for the first row this core handles

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    // constexpr uint32_t cb_reduce = tt::CBIndex::c_1; // No longer needed

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = get_tile_size(cb_inp);
    const DataFormat src0_data_format = get_dataformat(cb_inp);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t blk = 1;  // Read one tile at a time for simplicity with interleaved addressing
    // constexpr uint32_t dim_shard_factor = get_compile_time_arg_val(2);
    // constexpr uint32_t cores_per_dim_shard = get_compile_time_arg_val(3);

    const InterleavedAddrGenFast<src0_is_dram> src_a = {
        .bank_base_address = src_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};

    // Scalar generation removed, handled by merge stage

    uint32_t current_start_tile_id = start_tile_id;
    // DPRINT << "RL_LOOP_START num_rows=" << num_tile_rows << " tiles_per_dim=" << tiles_per_dim_shard << " Wt=" << Wt
    // << " Wt_off=" << Wt_offset << " start_id=" << start_tile_id << ENDL();

    // Loop through rows assigned to this core
    for (uint32_t row_idx = 0; row_idx < num_tile_rows; ++row_idx) {
        // Read only the tiles for this dimension shard within the current row
        uint32_t row_start_tile_id_for_shard = current_start_tile_id + Wt_offset;
        for (uint32_t wt = 0; wt < tiles_per_dim_shard; ++wt) {  // Iterate one tile at a time (blk=1)
            cb_reserve_back(cb_inp, blk);                        // Reserve for 1 tile
            // DPRINT << "RL_RESERVE_BACK R" << row_idx << " Wt" << wt << "/" << tiles_per_dim_shard << ENDL();
            uint32_t inp_wr_ptr = get_write_ptr(cb_inp);

            // Calculate the source tile ID for the current row and width position
            uint32_t src_tile_id = row_start_tile_id_for_shard + wt;
            // DPRINT << "  RD_TILE: " << src_tile_id << " -> CB" << cb_inp << "@" << inp_wr_ptr << ENDL();

            noc_async_read_tile(src_tile_id, src_a, inp_wr_ptr);
            noc_async_read_barrier();
            cb_push_back(cb_inp, blk);
            // DPRINT << "RL_PUSH_BACK" << ENDL();
        }
        // Move to the start of the next row in the interleaved tensor
        current_start_tile_id += Wt;
    }
    DPRINT << "RL_KRNL_END" << ENDL();
}
