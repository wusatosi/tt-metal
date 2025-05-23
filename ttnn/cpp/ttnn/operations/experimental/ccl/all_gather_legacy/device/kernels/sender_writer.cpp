// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>
#include <array>

FORCE_INLINE void send_packet(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes,
    uint32_t step) {
    const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
    const size_t payload_l1_address = l1_read_addr;

    volatile tt_l1_ptr uint16_t* dst_noc2 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_read_addr);
    for (uint16_t value = 0; value < 2; value++) {
        DPRINT << "value at " << (uint16_t)value << " is: " << BF16((uint16_t)dst_noc2[value]) << ENDL();
    }

    pkt_hdr_forward->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);

    if (step == 0) {
        noc_async_write(
            payload_l1_address, safe_get_noc_addr(dest_noc_xy.x, dest_noc_xy.y, dest_addr), payload_size_bytes);
    }
    if (fabric_connection.has_forward_connection()) {
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)pkt_hdr_forward, sizeof(PACKET_HEADER_TYPE));
    }
    noc_async_writes_flushed();

    l1_read_addr += payload_size_bytes;
}

void kernel_main() {
    constexpr uint32_t cb0_id = get_compile_time_arg_val(0);               // src buffer id
    const auto receiver_base_address = get_compile_time_arg_val(1);        // cb for receiver should be with fixed
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(2);  // packet header buffer id
    constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(4);
    constexpr uint32_t num_devices = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(6);  // num tiles per device
    constexpr uint32_t ring_index = get_compile_time_arg_val(7);

    DPRINT << "ct args: \n";
    DPRINT << "cb0_id: " << (uint32_t)cb0_id << "\n";
    DPRINT << "receiver_base_address: " << (uint32_t)receiver_base_address << "\n";
    DPRINT << "packet_header_cb_id: " << (uint32_t)packet_header_cb_id << "\n";
    DPRINT << "packet_size_in_pages: " << (uint32_t)packet_size_in_pages << "\n";
    DPRINT << "tensor0_page_size: " << (uint32_t)tensor0_page_size << "\n";
    DPRINT << "num_devices: " << (uint32_t)num_devices << "\n";
    DPRINT << "num_tiles: " << (uint32_t)num_tiles << "\n";
    DPRINT << "ring_index: " << (uint32_t)ring_index << "\n";

    size_t arg_idx = 0;
    const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);  // global semaphore address
                                                                              // address
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);    // nocx for receiver core
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);    // noc y for receiver core

    DPRINT << "rt args: \n";
    DPRINT << "out_ready_sem_bank_addr: " << (uint32_t)out_ready_sem_bank_addr << "\n";
    DPRINT << "out_ready_sem_noc0_x: " << (uint32_t)out_ready_sem_noc0_x << "\n";
    DPRINT << "out_ready_sem_noc0_y: " << (uint32_t)out_ready_sem_noc0_y << "\n";

    auto arg_for_fab = arg_idx;
    DPRINT << "fabric_connection arg 0" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 1" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 2" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 3" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 4" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";

    auto fabric_connection = FabricConnectionManager::build_from_args(arg_idx);

    // auto fabric_connection =
    //     FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
    //         arg_idx);
    DPRINT << "fabric connection built\n";
    // set up packet header buffer
    cb_reserve_back(packet_header_cb_id, 1);
    auto packet_header_buffer_addr_forward = get_write_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);
    cb_reserve_back(packet_header_cb_id, 1);
    auto packet_header_buffer_seminc = get_write_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    DPRINT << "after reserve_back\n";
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    pkt_hdr_forward->to_chip_unicast(1);  // 1 for num of hops
    DPRINT << "after pkt_hdr_forward\n";
    if (fabric_connection.is_logically_connected()) {
        DPRINT << "before open\n";
        fabric_connection.open();
        DPRINT << "after open\n";
    }

    for (uint32_t step = 0; step < num_devices; step++) {
        uint32_t start_tile_id = ((ring_index - step + num_devices) % num_devices) * num_tiles;
        DPRINT << "step: " << step << "\n";
        DPRINT << "start_tile_id: " << start_tile_id << "\n";
        uint64_t noc0_dest_noc_addr =
            get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, receiver_base_address, 0);
        uint32_t tile_id_end = start_tile_id + num_tiles;
        DPRINT << "tile_id_end: " << tile_id_end << "\n";
        uint32_t tile_id = start_tile_id;
        while (tile_id < tile_id_end) {
            DPRINT << "tile_id: " << tile_id << "\n";
            cb_wait_front(cb0_id, packet_size_in_pages);
            size_t l1_read_addr = get_read_ptr(cb0_id);
            uint32_t num_pages_to_read = std::min(tile_id_end - tile_id, packet_size_in_pages);
            DPRINT << "num_pages_to_read: " << num_pages_to_read << "\n";

            uint32_t contig_pages_advanced = 2;  // always 1 for interleaved
            for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                // write packet to destination
                send_packet(
                    noc0_dest_noc_addr,
                    pkt_hdr_forward,
                    fabric_connection,
                    l1_read_addr,
                    contig_pages_advanced * tensor0_page_size,
                    step);
                DPRINT << "after send_packet\n";
                noc0_dest_noc_addr += contig_pages_advanced * tensor0_page_size;
            }
            tile_id += num_pages_to_read;
            cb_pop_front(cb0_id, packet_size_in_pages);
        }

        // 2. unicast semaphore
        cb_reserve_back(packet_header_cb_id, 1);
        const uint32_t sem_header_addr = get_write_ptr(packet_header_cb_id);
        cb_push_back(packet_header_cb_id, 1);

        uint64_t out_ready_sem_noc_addr_in_pkt =
            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);
        auto* pkt_hdr = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
        pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            out_ready_sem_noc_addr_in_pkt,
            static_cast<uint16_t>(1),  // increment 1
            32});
        // Write the mcast packet (forward)
        if (fabric_connection.has_forward_connection()) {
            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
            pkt_hdr->to_chip_unicast(1);
            fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
        }
        DPRINT << "after semaphore send\n";
    }
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    noc_async_write_barrier();

    DPRINT << "DONE \n";
}
