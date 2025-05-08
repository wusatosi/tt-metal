// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <cstddef>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;

constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t cb1_id = get_compile_time_arg_val(1);
constexpr uint32_t data_size = get_compile_time_arg_val(2);

void write_and_advance_local_read_address_for_fabric_write(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes) {
    DPRINT << "1.\n";
    const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
    const size_t payload_l1_address = l1_read_addr;
    DPRINT << "2.\n";
    pkt_hdr_forward->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);
    DPRINT << "3.\n";
    // pkt_hdr_backward->to_noc_unicast_write(
    //     tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);

    noc_async_write(payload_l1_address, safe_get_noc_addr(dest_noc_xy.x, dest_noc_xy.y, dest_addr), payload_size_bytes);

    if (fabric_connection.has_forward_connection()) {
        DPRINT << "4.\n";
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        DPRINT << "5.\n";
        fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        DPRINT << "6.\n";
        fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)pkt_hdr_forward, sizeof(PACKET_HEADER_TYPE));
        DPRINT << "7.\n";
    }
    if (fabric_connection.has_backward_connection()) {
        DPRINT << "8.\n";
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        DPRINT << "9.\n";
        fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        DPRINT << "10.\n";
        fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)pkt_hdr_backward, sizeof(PACKET_HEADER_TYPE));
        DPRINT << "11.\n";
    }
    noc_async_writes_flushed();

    l1_read_addr += payload_size_bytes;
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    uint32_t num_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t reserved_packet_header_cb_id = get_arg_val<uint32_t>(arg_idx++);

    DPRINT << "HELLO FROM READER\n";

    auto fabric_connection =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);

    DPRINT << "BUILT FABRIC CONNECTION\n";

    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_forward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    DPRINT << "FIRST HEADER DONE\n";
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_backward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    DPRINT << "SECOND HEADER DONE\n";
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    DPRINT << "THIRD HEADER DONE\n";

    DPRINT << "CREATE PACKETS COMPLETE\n";

    // Create headers
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);

    DPRINT << "CREATE HEADERS COMPLETE\n";
    pkt_hdr_forward->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, 0});
    pkt_hdr_backward->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, 0});

    DPRINT << "pkt_hdr_forward: " << (uint32_t)pkt_hdr_forward << "\n";
    DPRINT << "pkt_hdr_backward: " << (uint32_t)pkt_hdr_backward << "\n";

    DPRINT << "rt args: \n";
    DPRINT << "tensor_address0: " << (uint32_t)tensor_address0 << "\n";
    DPRINT << "num_tiles: " << (uint32_t)num_tiles << "\n";

    // interleaved addrgen
    uint32_t tensor0_page_size = 1088;
    auto tensor0_addrgen = InterleavedAddrGenFast<true>{
        .bank_base_address = tensor_address0, .page_size = tensor0_page_size, .data_format = get_dataformat(cb0_id)};

    uint32_t l1_write_addr = get_write_ptr(cb1_id);

    auto dst_addrgen = InterleavedAddrGenFast<false>{
        .bank_base_address = l1_write_addr, .page_size = tensor0_page_size, .data_format = get_dataformat(cb1_id)};

    DPRINT << "tensor -> CB: " << (uint32_t)cb0_id << "\n";
    DPRINT << "data size: " << (uint32_t)data_size << "\n";

    uint32_t tile_id = 0;

    DPRINT << "DONE OPENING FABRIC CONNECTION\n";

    for (tile_id = 0; tile_id < num_tiles; tile_id++) {
        DPRINT << "tile_id: " << tile_id << "\t" << "num_tiles: " << (uint32_t)num_tiles << "\n";
        cb_reserve_back(cb0_id, 1);
        uint32_t l1_read_addr = get_write_ptr(cb0_id);
        noc_async_read_tile(tile_id, tensor0_addrgen, l1_read_addr);
        DPRINT << "got noc_async_read\n";
        // l1_read_addr += data_size;
        // noc_async_read_barrier();
        DPRINT << "got noc_async_read_barrier\n";
        cb_push_back(cb0_id, 1);
        DPRINT << "got cb_push_back\n";

        size_t l1_read_addr_t = static_cast<size_t>(l1_read_addr);

        size_t dst_l1_noc_addr = dst_addrgen.get_noc_addr(tile_id);

        DPRINT << "WRITING TO ETH\n";
        write_and_advance_local_read_address_for_fabric_write(
            dst_l1_noc_addr, pkt_hdr_forward, pkt_hdr_backward, fabric_connection, l1_read_addr_t, data_size);

        DPRINT << "WRITING TO ETH DONE\n";
        DPRINT << "Incrementing semaphore\n";

        // uint64_t out_ready_sem_noc_addr_in_pkt =
        // safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);
        // auto* pkt_hdr = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
        // pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        //     out_ready_sem_noc_addr_in_pkt,
        //     static_cast<uint16_t>(1),  // increment 1
        //     32});
        // // Write the mcast packet (forward)
        // fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        // pkt_hdr->to_chip_multicast(
        //     tt::tt_fabric::MulticastRoutingCommandHeader{1, 0});
        // fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
        //     packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }

    DPRINT << "DONE WITH READER\n";
}
