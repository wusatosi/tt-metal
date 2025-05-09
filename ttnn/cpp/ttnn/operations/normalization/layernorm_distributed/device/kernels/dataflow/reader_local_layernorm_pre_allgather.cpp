// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel reads the layernorm inputs from interleaved dram for the local computation stage.
 * It handles dimension sharding by reading only the subset of dimension tiles assigned to this core.
 */

#include <stdint.h>
#include "dataflow_api.h"
#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "debug/dprint.h"
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);             // Source address in dram
    const uint32_t NCHt = get_arg_val<uint32_t>(1);                 // Number of NCH tiles
    const uint32_t tiles_per_dim_shard = get_arg_val<uint32_t>(2);  // Number of tiles in this dimension shard
    const uint32_t tile_offset = get_arg_val<uint32_t>(3);          // Tile offset for this core
    const uint32_t reduce_scalar = get_arg_val<uint32_t>(4);        // Reduce scalar value

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = get_tile_size(cb_inp);
    const DataFormat src0_data_format = get_dataformat(cb_inp);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t dim_shard_factor = get_compile_time_arg_val(2);
    constexpr uint32_t cores_per_dim_shard = get_compile_time_arg_val(3);

    const InterleavedAddrGenFast<src0_is_dram> src_a = {
        .bank_base_address = src_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};

    DPRINT << "RL_START" << ENDL();

    // Generate constant tiles for reduce scalar
    generate_reduce_scaler(cb_reduce, reduce_scalar);

    DPRINT << "RL_REDUCE_SCALAR_GENERATED" << ENDL();

    // Calculate the starting tile index for this core
    uint32_t inp_tile_idx = tile_offset;

    for (uint32_t ncht = 0; ncht < NCHt; ++ncht) {
        // Read only the tiles for this dimension shard
        for (uint32_t wt = 0; wt < tiles_per_dim_shard; wt += blk) {
            cb_reserve_back(cb_inp, blk);
            DPRINT << "RL_RESERVE_BACK" << wt << "/" << tiles_per_dim_shard << ENDL();
            uint32_t inp_wr_ptr = get_write_ptr(cb_inp);

            for (uint32_t r = 0; r < blk; r++) {
                noc_async_read_tile(inp_tile_idx, src_a, inp_wr_ptr);
                inp_wr_ptr += src0_tile_bytes;
                inp_tile_idx++;
            }
            noc_async_read_barrier();
            cb_push_back(cb_inp, blk);
            DPRINT << "RL_PUSH_BACK" << inp_tile_idx << "blk" << blk << ENDL();
        }
    }
}
