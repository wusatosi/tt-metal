// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_constants.hpp>
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_ccl_common.hpp"
#include "cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/transaction_id_tracker.hpp"
#include <cstdint>
#include <utility>
#include <array>
using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(2);
constexpr uint32_t cb0_id = get_compile_time_arg_val(3);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(4);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(5);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(6);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(7);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    uint32_t reduction_input_cb_id = get_arg_val<address_t>(arg_idx++);
    address_t reduction_input_addr = get_write_ptr(reduction_input_cb_id);

    const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_core_tile_start_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_mcast_cores = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t reduction_semaphore_send_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t num_mcast_ranges = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t link = get_arg_val<uint32_t>(arg_idx++);

    // Set up for mcasting to reduction workers
    volatile tt_l1_ptr uint32_t* reduction_semaphore_send_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduction_semaphore_send_addr);
    noc_semaphore_set(reduction_semaphore_send_addr_ptr, VALID);

    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;

    tt_l1_ptr uint32_t* mcast_dest_noc_start_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_mcast_ranges;
    tt_l1_ptr uint32_t* mcast_dest_noc_start_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_mcast_ranges;
    tt_l1_ptr uint32_t* mcast_dest_noc_end_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_mcast_ranges;
    tt_l1_ptr uint32_t* mcast_dest_noc_end_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_mcast_ranges;

    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_idx);

    constexpr uint32_t NUM_TRANSACTION_IDS = num_packet_headers_storable / 2;
    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, NUM_TRANSACTION_IDS);
    auto packet_header_buffer_addr_forward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, NUM_TRANSACTION_IDS);
    cb_reserve_back(reserved_packet_header_cb_id, NUM_TRANSACTION_IDS);
    auto packet_header_buffer_addr_backward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, NUM_TRANSACTION_IDS);
    cb_reserve_back(reserved_packet_header_cb_id, NUM_TRANSACTION_IDS);
    auto packet_header_buffer_seminc = packet_header_buffer_addr_forward;

    // pre-populate packet headers
    std::array<volatile PACKET_HEADER_TYPE*, NUM_TRANSACTION_IDS> pkt_hdrs_forward;
    std::array<volatile PACKET_HEADER_TYPE*, NUM_TRANSACTION_IDS> pkt_hdrs_backward;
    for (uint32_t i = 0; i < NUM_TRANSACTION_IDS; i++) {
        pkt_hdrs_forward[i] = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(
            packet_header_buffer_addr_forward + i * sizeof(PACKET_HEADER_TYPE));
        pkt_hdrs_backward[i] = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(
            packet_header_buffer_addr_backward + i * sizeof(PACKET_HEADER_TYPE));
        pkt_hdrs_forward[i]->to_chip_multicast(
            tt::fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
        pkt_hdrs_backward[i]->to_chip_multicast(
            tt::fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }

    // 1. mcast via fabric to remote tensor addresses
    uint32_t tiles_read = 0;
    uint32_t shard_tile_id = first_core_tile_start_offset;
    uint32_t core_id = 0;
    uint32_t writer_chip_offset = my_chip_id * num_tiles_per_core * tensor0_page_size;

    // Transaction ID related constants/types
    WriteTransactionIdTracker<NUM_TRANSACTION_IDS> trid_tracker(cb0_id);

    while (tiles_read < num_tiles_to_read) {
        uint32_t num_tiles_to_read_this_core = std::min(num_tiles_per_core - shard_tile_id, packet_size_in_pages);
        num_tiles_to_read_this_core = std::min(num_tiles_to_read - tiles_read, num_tiles_to_read_this_core);
        if (trid_tracker.has_unflushed_trid() && trid_tracker.oldest_trid_flushed()) {
            DPRINT << "oldest_trid_flushed\n";
            trid_tracker.pop_pages_for_oldest_trid();
        }
        if (!trid_tracker.backpressured() && trid_tracker.cb_next_slot_is_available(num_tiles_to_read_this_core)) {
            DPRINT << "cb_next_slot_is_available\n";
            auto [l1_read_addr, trid] = trid_tracker.get_next_cb_slot(num_tiles_to_read_this_core);

            uint64_t noc0_dest_noc_addr =
                get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], reduction_input_addr + writer_chip_offset);

            // Within-shard offset
            noc0_dest_noc_addr += shard_tile_id * tensor0_page_size;

            // trid also used to index packet headers
            write_and_advance_local_read_address_for_fabric_write(
                noc0_dest_noc_addr,
                pkt_hdrs_forward[trid.get()],
                pkt_hdrs_backward[trid.get()],
                fabric_connection,
                l1_read_addr,
                num_tiles_to_read_this_core * tensor0_page_size,
                trid);

            tiles_read += num_tiles_to_read_this_core;
            shard_tile_id += num_tiles_to_read_this_core;
            if (shard_tile_id >= num_tiles_per_core) {
                shard_tile_id = 0;
                core_id++;
            }
        }
    }
    DPRINT << "DONE MAIN LOOP\n";
    // noc_async_writes_flushed();
    trid_tracker.write_barrier();
    DPRINT << "DONE write barrier\n";

    // 2. mcast output ready semaphore
    // DO NOT MOVE THIS INITIALIZATION UP - MEMORY ALIASED WITH FORWARD PACKET HEADER
    auto* pkt_hdr = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
    pkt_hdr->to_noc_unicast_atomic_inc(tt::fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_noc_addr_in_pkt,
        static_cast<uint16_t>(1),  // increment 1
        32});
    // Write the mcast packet (forward)
    if (fabric_connection.has_forward_connection()) {
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        pkt_hdr->to_chip_multicast(
            tt::fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
        fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }
    // Write the mcast packet (backward)
    if (fabric_connection.has_backward_connection()) {
        pkt_hdr->to_chip_multicast(
            tt::fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }
    // increment locally
    uint64_t out_ready_sem_noc_addr =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
    noc_semaphore_inc(out_ready_sem_noc_addr, 1);

    DPRINT << "Wait data ready\n";
    // 3. wait for mcast output ready semaphore
    while (*reinterpret_cast<volatile uint32_t*>(out_ready_sem_bank_addr) != out_ready_sem_wait_value);
    DPRINT << "Data ready\n";

    DPRINT << "Notify reducer\n";
    // loop over mcast ranges
    for (uint32_t i = 0; i < num_mcast_ranges; i++) {
        // Signal the reduction workers
        const uint64_t reduction_semaphore_recv_noc_addr = get_noc_multicast_addr(
            mcast_dest_noc_start_x[i],
            mcast_dest_noc_start_y[i],
            mcast_dest_noc_end_x[i],
            mcast_dest_noc_end_y[i],
            reduction_semaphore_send_addr);

        noc_semaphore_set_multicast(
            reduction_semaphore_send_addr,
            reduction_semaphore_recv_noc_addr,
            i == 0 ? num_mcast_cores : 0,
            false,  // linked = false
            true);  // multicast_path_reserve = true
    }

    // 4. global semaphore reset
    *reinterpret_cast<volatile uint32_t*>(out_ready_sem_bank_addr) = 0;

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    trid_tracker.write_barrier();
    noc_async_write_barrier();

    // DPRINT << "writer done \n";
}
