// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
// #include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp" // Not needed if using noc_ directly
#include "debug/dprint.h"  // Re-add for DPRINT

// Args (RUNTIME):
// 0: packed_reduce_scaler
// Args (COMPILE TIME)
// 0: is_rmsnorm
// 1: num_dims_per_group
// 2: receiver_sem_compile_idx
// 3: sender_sem_compile_idx
// 4: writer_noc_start_x
// 5: writer_noc_start_y
// 6: writer_noc_end_x
// 7: writer_noc_end_y
// 8: num_dests (num writers)
void kernel_main() {
    // Runtime Args
    const uint32_t packed_reduce_scaler = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr bool is_rmsnorm = get_compile_time_arg_val(0);
    constexpr uint32_t num_dims_per_group = get_compile_time_arg_val(1);
    constexpr uint32_t receiver_sem_compile_idx = get_compile_time_arg_val(2);
    constexpr uint32_t sender_sem_compile_idx = get_compile_time_arg_val(3);
    constexpr uint32_t writer_noc_start_x = get_compile_time_arg_val(4);
    constexpr uint32_t writer_noc_start_y = get_compile_time_arg_val(5);
    constexpr uint32_t writer_noc_end_x = get_compile_time_arg_val(6);
    constexpr uint32_t writer_noc_end_y = get_compile_time_arg_val(7);
    constexpr uint32_t num_dests = get_compile_time_arg_val(8);

    // Get semaphore L1 addresses using compile-time indices
    uint32_t receiver_sem_addr = get_semaphore(receiver_sem_compile_idx);
    uint32_t sender_sem_addr = get_semaphore(sender_sem_compile_idx);

    // Cast L1 addresses to pointers for noc_semaphore_wait/set
    volatile tt_l1_ptr uint32_t* receiver_sem_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr);
    volatile tt_l1_ptr uint32_t* sender_sem_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem_addr);

    // Get multicast NOC address for receiver semaphore
    uint64_t mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        writer_noc_start_x, writer_noc_start_y, writer_noc_end_x, writer_noc_end_y, receiver_sem_addr);

    // CBs
    constexpr uint32_t merge_data_cb_id = tt::CBIndex::c_9;
    constexpr uint32_t compute_in_cb_id = tt::CBIndex::c_1;  // CB for scaler
    // constexpr uint32_t compute_out_cb_id = tt::CBIndex::c_14; // Not used by reader
    constexpr uint32_t num_tiles_per_shard = is_rmsnorm ? 1 : 2;
    constexpr uint32_t total_tiles_to_read = num_dims_per_group * num_tiles_per_shard;
    // const uint32_t single_tile_size = get_tile_size(merge_data_cb_id); // Not needed by reader

    // --- Semaphore setup phase (Mimic TopK reader) ---
    // Ensure sender starts at 0
    DPRINT << "RF_SET_SENDER_INIT sem_addr=" << sender_sem_addr << ENDL();
    noc_semaphore_set(sender_sem_addr_ptr, 0);
    // Set local receiver to VALID and multicast to writers
    DPRINT << "RF_SET_RECEIVER_LOCAL_VALID sem_addr=" << receiver_sem_addr << ENDL();
    noc_semaphore_set(receiver_sem_addr_ptr, 1);
    DPRINT << "RF_MULTICAST_RECEIVER_VALID mcast_addr=" << mcast_receiver_semaphore_noc_addr
           << " num_dests=" << num_dests << ENDL();
    noc_semaphore_set_multicast(receiver_sem_addr, mcast_receiver_semaphore_noc_addr, num_dests);
    DPRINT << "RF_SEMAPHORE_INIT_DONE" << ENDL();

    // --- Main Loop/Logic --- // Assuming one iteration for simplicity now

    // Prepare scaler for compute kernel
    cb_reserve_back(compute_in_cb_id, 1);
    auto scaler_tile_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(compute_in_cb_id));
    scaler_tile_ptr[0] = packed_reduce_scaler & 0xFFFF;
    cb_push_back(compute_in_cb_id, 1);
    DPRINT << "RF_REDUCE_SCALAR_SET inv_N=" << packed_reduce_scaler << ENDL();

    // Wait for all writers to signal they are done (increment sender semaphore)
    DPRINT << "RF_WAIT_SENDERS target=" << num_dims_per_group << " sem_addr=" << sender_sem_addr << ENDL();
    noc_semaphore_wait(sender_sem_addr_ptr, num_dims_per_group);
    DPRINT << "RF_SENDERS_DONE val=" << *sender_sem_addr_ptr << ENDL();
    cb_push_back(merge_data_cb_id, total_tiles_to_read);

    // Data is ready in merge CB, pop it (assuming compute doesn't need it persisted)
    // cb_wait_front(merge_data_cb_id, total_tiles_to_read);
    // cb_pop_front(merge_data_cb_id, total_tiles_to_read);
    // DPRINT << "RF_POP_MERGE_CB tiles=" << total_tiles_to_read << ENDL();

    // Reset semaphores for potential next iteration
    // DPRINT << "RF_SET_RECEIVER_INVALID sem_addr=" << receiver_sem_addr << ENDL();
    // noc_semaphore_set(receiver_sem_addr_ptr, 0);
    // DPRINT << "RF_SET_RECEIVER_DONE val=" << *receiver_sem_addr_ptr << ENDL();

    // DPRINT << "RF_SET_SENDER_INVALID sem_addr=" << sender_sem_addr << ENDL();
    // noc_semaphore_set(sender_sem_addr_ptr, 0);
    // DPRINT << "RF_SET_SENDER_DONE val=" << *sender_sem_addr_ptr << ENDL();

    DPRINT << "RF_KRNL_END" << ENDL();
}  // kernel_main()
