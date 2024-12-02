// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "debug/dprint.h"

void kernel_main() {
    // Runtime args
    uint32_t rt_args_idx = 0;
    uint32_t noc = get_arg_val<uint32_t>(rt_args_idx++);
    bool master_reducer = (bool)get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t reducer_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t master_reducer_core_noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t master_reducer_core_noc_y = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t master_reducer_core_noc = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t* slave_reducer_cores_noc = (uint32_t*)get_arg_addr(rt_args_idx++);

    // Compile time args
    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t batch = get_compile_time_arg_val(2);

    constexpr uint32_t num_reducer_partials = get_compile_time_arg_val(3);
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(4);
    const uint32_t reducer_semaphore_addr = get_semaphore(get_compile_time_arg_val(5));
    volatile tt_l1_ptr uint32_t* l1_reducer_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reducer_semaphore_addr);

    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t out_cb_id = tt::CB::c_out0;
    constexpr uint32_t reducer_cb_id = tt::CB::c_intermed1;
    constexpr uint32_t partial_cb_id = tt::CB::c_intermed2;

    constexpr uint32_t shard_size_in_tiles = shard_width_in_tiles * shard_height_in_tiles;
    constexpr uint32_t out_block_size_in_bytes = out_block_num_tiles * get_tile_size(reducer_cb_id);

    cb_reserve_back(cb_id_in1, shard_size_in_tiles);
    cb_push_back(cb_id_in1, shard_size_in_tiles);

    for (uint32_t b = 0; b < batch; ++b) {
        if (!master_reducer) {
            /*
            1. Wait front on the tiles in the partial output buffer.
                Here, the wait front must be at the same granularity as the out_block_num_tiles (for now)
            2. noc_async_write the tiles to the reducer core
            3. noc_semaphore_inc the semaphore on the reducer core
            */

            uint64_t remote_reducer_semaphore_addr = get_noc_addr(
                master_reducer_core_noc_x, master_reducer_core_noc_y, reducer_semaphore_addr, master_reducer_core_noc);
            // NOTE: Using read ptr as address to the start of the buffer is needed
            uint32_t local_l1_write_addr =
                get_write_ptr(reducer_cb_id) +
                out_block_size_in_bytes *
                    (reducer_idx - 1);  // reducer_idx is 1-indexed for slave reducers, but the buffer is 0-indexed
            uint64_t remote_l1_write_addr = get_noc_addr(
                master_reducer_core_noc_x, master_reducer_core_noc_y, local_l1_write_addr, master_reducer_core_noc);

            cb_wait_front(partial_cb_id, out_block_num_tiles);

            // Wait for go-ahead signal from master reducer
            noc_semaphore_wait_min(l1_reducer_sem_addr, b);

            noc_async_write(
                get_read_ptr(partial_cb_id), remote_l1_write_addr, out_block_size_in_bytes, master_reducer_core_noc);
            noc_semaphore_inc(remote_reducer_semaphore_addr, 1, master_reducer_core_noc);
            cb_pop_front(partial_cb_id, out_block_num_tiles);

        } else if (num_reducer_partials > 1) {
            /*
            1. noc_semaphore wait on the semaphore to wait for all incoming data to reduce
            2. push_back on the correct amount of data in the reducer_cb,
                so that the compute kernel can start the reduction process
            */

            noc_semaphore_wait_min(l1_reducer_sem_addr, num_reducer_partials - 1);
            noc_semaphore_set(l1_reducer_sem_addr, 0);

            cb_reserve_back(reducer_cb_id, out_block_num_tiles * (num_reducer_partials - 1));
            cb_push_back(reducer_cb_id, out_block_num_tiles * (num_reducer_partials - 1));

            // Wait to finish reduction
            cb_wait_front(out_cb_id, out_block_num_tiles * (b + 1));
            // TODO: Send all reduce fusion signal
            cb_pop_front(reducer_cb_id, out_block_num_tiles * (num_reducer_partials - 1));

            // Send the go-ahead signal to the slave cores
            for (uint32_t p = 0; p < num_reducer_partials - 1; p++) {  // TODO: Maybe use noc_mcast_semaphore_inc
                uint32_t core_noc_x = slave_reducer_cores_noc[p * 3];
                uint32_t core_noc_y = slave_reducer_cores_noc[p * 3 + 1];
                uint32_t core_noc = slave_reducer_cores_noc[p * 3 + 2];

                uint64_t remote_reducer_semaphore_addr =
                    get_noc_addr(core_noc_x, core_noc_y, reducer_semaphore_addr, core_noc);
                noc_semaphore_inc(remote_reducer_semaphore_addr, 1, core_noc);
            }

            // NOTE: Future optimization: change the granularity for the the wait and push
        }
    }
}
