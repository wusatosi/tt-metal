#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // Get runtime arguments
    const uint32_t noc_final_core_x = get_arg_val<uint32_t>(0);  // X coordinate of merge core
    const uint32_t noc_final_core_y = get_arg_val<uint32_t>(1);  // Y coordinate of merge core
    const uint32_t dimension_id = get_arg_val<uint32_t>(2);      // Dimension ID
    const uint32_t base_offset = get_arg_val<uint32_t>(3);       // Base offset in merge buffer

    DPRINT << "WL_START" << ENDL();

    // Get compile time arguments
    constexpr uint32_t semaphore_id = get_compile_time_arg_val(0);  // Semaphore ID base
    constexpr bool is_rmsnorm = get_compile_time_arg_val(1);
    constexpr uint32_t dim_shard_factor = get_compile_time_arg_val(2);
    constexpr uint32_t cores_per_dim = get_compile_time_arg_val(3);

    // Circular buffer indices
    constexpr uint32_t partial_cb_index = tt::CBIndex::c_7;     // Partial results from compute
    constexpr uint32_t merge_data_cb_index = tt::CBIndex::c_9;  // Data buffer at merge core

    // Calculate number of results per dimension
    uint32_t results_per_dim = is_rmsnorm ? 1 : 2;

    DPRINT << "WL_INIT_COMPLETE" << ENDL();

    // Sizes for transfers
    constexpr uint32_t onetile = 1;
    uint32_t partial_tile_bytes = get_tile_size(partial_cb_index);

    // Get the remote semaphore addresses at the merge core
    uint32_t receiver_semaphore = get_semaphore(semaphore_id);
    uint32_t sender_semaphore = get_semaphore(semaphore_id + 1);

    uint64_t noc_receiver_semaphore_addr = get_noc_addr(noc_final_core_x, noc_final_core_y, receiver_semaphore);
    uint64_t noc_sender_semaphore_addr = get_noc_addr(noc_final_core_x, noc_final_core_y, sender_semaphore);

    DPRINT << "WL_SEMAPHORES_OBTAINED" << ENDL();

    // Calculate write location in merge buffer
    // Start at the base_offset for this sequence
    // Add offset for this dimension's results
    uint32_t dimension_offset = dimension_id * results_per_dim * partial_tile_bytes;
    uint32_t dest_addr = base_offset + dimension_offset;
    uint64_t noc_merge_data_addr = get_noc_addr(noc_final_core_x, noc_final_core_y, dest_addr);

    // Wait for the partial results from the compute kernel
    cb_wait_front(partial_cb_index, results_per_dim);

    DPRINT << "WL_PARTIAL_RESULTS_RECEIVED" << ENDL();

    // Wait until the receiver semaphore is VALID (merge core is ready to receive)
    volatile tt_l1_ptr uint32_t* local_receiver_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(semaphore_id));
    noc_semaphore_wait(local_receiver_semaphore, VALID);

    DPRINT << "WL_RECEIVER_SEMAPHORE_VALID" << ENDL();

    // Write E(x^2) partial result to merge core
    uint32_t partial_read_addr = get_read_ptr(partial_cb_index);
    noc_async_write(partial_read_addr, noc_merge_data_addr, partial_tile_bytes);
    cb_pop_front(partial_cb_index, 1);

    DPRINT << "WL_EX2_WRITTEN" << ENDL();

    // If layernorm, also write E(x) partial result
    if (!is_rmsnorm) {
        partial_read_addr = get_read_ptr(partial_cb_index);
        noc_async_write(partial_read_addr, noc_merge_data_addr + partial_tile_bytes, partial_tile_bytes);
        cb_pop_front(partial_cb_index, 1);
        DPRINT << "WL_EX_WRITTEN" << ENDL();
    }

    // Ensure all writes complete before signaling
    noc_async_write_barrier();

    DPRINT << "WL_WRITE_BARRIER_COMPLETE" << ENDL();

    // Signal to the merge core that this dimension has completed
    noc_semaphore_inc(noc_sender_semaphore_addr, 1);

    DPRINT << "WL_COMPLETE" << ENDL();
}
