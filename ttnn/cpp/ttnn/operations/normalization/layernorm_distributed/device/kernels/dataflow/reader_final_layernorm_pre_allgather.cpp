#include <stdint.h>
#include "dataflow_api.h"
#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "debug/dprint.h"

void kernel_main() {
    // Read runtime arguments
    const uint32_t semaphore_id = get_compile_time_arg_val(0);  // Semaphore ID for synchronization
    const uint32_t packed_inv_N = get_arg_val<uint32_t>(0);     // 1/N normalization factor

    DPRINT << "RF_START" << ENDL();

    // Get compile time arguments
    constexpr bool is_rmsnorm = get_compile_time_arg_val(1);
    constexpr uint32_t dim_shard_factor = get_compile_time_arg_val(2);
    constexpr uint32_t noc_start_x = get_compile_time_arg_val(3);
    constexpr uint32_t noc_start_y = get_compile_time_arg_val(4);
    constexpr uint32_t noc_end_x = get_compile_time_arg_val(5);
    constexpr uint32_t noc_end_y = get_compile_time_arg_val(6);
    constexpr uint32_t num_dests = get_compile_time_arg_val(7);

    // Circular buffer indices
    constexpr uint32_t merge_data_cb_index = tt::CBIndex::c_9;  // Data for merge computation
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;            // Reduction scalar buffer

    // Calculate number of partial results to read based on dimension sharding
    const uint32_t results_per_dim = is_rmsnorm ? 1 : 2;
    const uint32_t total_partial_tiles = dim_shard_factor * results_per_dim;

    DPRINT << "RF_INIT_COMPLETE" << ENDL();

    // Initialize the normalization factor (1/N) in the reduce buffer
    // This will be used by the compute kernel
    generate_reduce_scaler(cb_reduce, packed_inv_N);

    DPRINT << "RF_REDUCE_SCALAR_SET" << ENDL();

    // Get the semaphore addresses
    uint32_t receiver_semaphore = get_semaphore(semaphore_id);
    uint32_t sender_semaphore = get_semaphore(semaphore_id + 1);  // Use next semaphore ID for sender

    volatile tt_l1_ptr uint32_t* receiver_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore);
    volatile tt_l1_ptr uint32_t* sender_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore);

    DPRINT << "RF_SEMAPHORES_OBTAINED" << ENDL();

    // Create multicast address for broadcasting to dimension shard cores
    uint64_t mcast_receiver_semaphore_noc_addr =
        get_noc_multicast_addr(noc_start_x, noc_start_y, noc_end_x, noc_end_y, receiver_semaphore);

    // Reserve space in merge data buffer
    cb_reserve_back(merge_data_cb_index, total_partial_tiles);

    DPRINT << "RF_BUFFER_RESERVED" << ENDL();

    // Mark sender semaphore as INVALID until we've received all data
    noc_semaphore_set(sender_semaphore_addr, INVALID);

    // Set the receiver semaphore to VALID to allow the senders to write
    noc_semaphore_set(receiver_semaphore_addr, VALID);

    DPRINT << "RF_SEMAPHORES_INITIALIZED" << ENDL();

    // Update the multicast address for the receiver semaphore
    noc_semaphore_set_multicast(receiver_semaphore, mcast_receiver_semaphore_noc_addr, num_dests);

    DPRINT << "RF_MULTICAST_SET" << ENDL();

    // Wait for all dimension shards to send their data
    DPRINT << "RF_WAITING_FOR_SHARDS" << ENDL();
    noc_semaphore_wait(sender_semaphore_addr, dim_shard_factor);

    DPRINT << "RF_ALL_SHARDS_RECEIVED" << ENDL();

    // Push back all tiles for the compute kernel to process
    cb_push_back(merge_data_cb_index, total_partial_tiles);

    // Reset the semaphores for the next sequence
    noc_semaphore_set(receiver_semaphore_addr, INVALID);
    noc_semaphore_set(sender_semaphore_addr, INVALID);

    DPRINT << "RF_COMPLETE" << ENDL();
}
