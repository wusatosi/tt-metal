// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
// #include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp" // Not needed
#include "debug/dprint.h"  // Re-add DPRINT

// Args (RUNTIME):
// 0: merge_core_noc_x
// 1: merge_core_noc_y
// 2: dim_shard_idx
// 3: remote_merge_cb_base_addr
// Args (COMPILE TIME):
// 0: is_rmsnorm
// 1: num_dims_per_group
// 2: dim_shard_factor
// 3: receiver_sem_compile_idx
// 4: sender_sem_compile_idx
void kernel_main() {
    // Runtime Args
    uint32_t merge_core_noc_x = get_arg_val<uint32_t>(0);
    uint32_t merge_core_noc_y = get_arg_val<uint32_t>(1);
    uint32_t dim_shard_idx = get_arg_val<uint32_t>(2);
    uint32_t remote_merge_cb_base_addr = get_arg_val<uint32_t>(3);

    // Compile time args
    constexpr bool is_rmsnorm = get_compile_time_arg_val(0);
    constexpr uint32_t num_dims_per_group = get_compile_time_arg_val(1);
    constexpr uint32_t dim_shard_factor = get_compile_time_arg_val(2);
    constexpr uint32_t receiver_sem_compile_idx = get_compile_time_arg_val(3);
    constexpr uint32_t sender_sem_compile_idx = get_compile_time_arg_val(4);

    // Get semaphore L1 addresses using compile-time indices
    uint32_t receiver_sem_addr = get_semaphore(receiver_sem_compile_idx);
    uint32_t sender_sem_addr = get_semaphore(sender_sem_compile_idx);

    // Cast L1 addresses to pointers for noc_semaphore_wait/set
    volatile tt_l1_ptr uint32_t* receiver_sem_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr);
    // volatile tt_l1_ptr uint32_t* sender_sem_addr_ptr   = reinterpret_cast<volatile tt_l1_ptr
    // uint32_t*>(sender_sem_addr); // Sender pointer not needed locally

    // Get remote NOC address for sender semaphore using the L1 address
    uint64_t remote_sender_sem_noc_addr = get_noc_addr(merge_core_noc_x, merge_core_noc_y, sender_sem_addr);

    // CBs
    constexpr uint32_t partial_results_cb_id = tt::CBIndex::c_7;
    constexpr uint32_t merge_data_cb_id = tt::CBIndex::c_9;  // Not used locally, only for remote address calculation
    constexpr uint32_t single_tile_size = get_tile_size(partial_results_cb_id);
    constexpr uint32_t num_tiles_per_shard = is_rmsnorm ? 1 : 2;

    // Destination address setup
    uint64_t merge_core_base_noc_addr = get_noc_addr(merge_core_noc_x, merge_core_noc_y, remote_merge_cb_base_addr);

    // Wait for receiver (on merge core) to signal ready via multicast
    DPRINT << "WL_WAIT_RECEIVER sem_addr=" << receiver_sem_addr << ENDL();
    noc_semaphore_wait(receiver_sem_addr_ptr, 1);  // Wait for VALID (1)
    DPRINT << "WL_RECEIVER_READY val=" << *receiver_sem_addr_ptr << ENDL();

    // Write partial results to merge core
    cb_wait_front(partial_results_cb_id, num_tiles_per_shard);
    uint32_t l1_read_addr = get_read_ptr(partial_results_cb_id);
    uint64_t dst_noc_addr = merge_core_base_noc_addr + (dim_shard_idx * num_tiles_per_shard * single_tile_size);
    DPRINT << "WL_WRITE shard=" << dim_shard_idx << " to (" << merge_core_noc_x << "," << merge_core_noc_y
           << ") addr=0x" << remote_merge_cb_base_addr
           << " offset=" << dim_shard_idx * num_tiles_per_shard * single_tile_size << ENDL();
    noc_async_write(l1_read_addr, dst_noc_addr, num_tiles_per_shard * single_tile_size);
    noc_async_write_barrier();
    cb_pop_front(partial_results_cb_id, num_tiles_per_shard);

    // Signal sender done by incrementing semaphore on merge core
    DPRINT << "WL_INC_SENDER noc_addr=" << remote_sender_sem_noc_addr << " (based on addr=" << sender_sem_addr << ")"
           << ENDL();
    noc_semaphore_inc(remote_sender_sem_noc_addr, 1);
    DPRINT << "WL_SENDER_SIGNALED" << ENDL();

    DPRINT << "WL_KRNL_END" << ENDL();
}  // kernel_main()
