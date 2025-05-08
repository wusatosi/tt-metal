// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <cstddef>
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;

constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t cb1_id = get_compile_time_arg_val(1);
constexpr uint32_t data_size = get_compile_time_arg_val(2);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    uint32_t num_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t reserved_packet_header_cb_id = get_arg_val<uint32_t>(arg_idx++);

    uint32_t semaphore = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);

    auto fabric_connection =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);

    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_forward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);

    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_backward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);

    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);

    // Create headers
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);

    pkt_hdr_forward->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, 0});
    pkt_hdr_backward->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, 0});

    // interleaved addrgen
    uint32_t tensor0_page_size = 1088;
    auto tensor0_addrgen = InterleavedAddrGenFast<true>{
        .bank_base_address = tensor_address0, .page_size = tensor0_page_size, .data_format = get_dataformat(cb0_id)};

    uint32_t l1_write_addr = get_write_ptr(cb1_id);

    auto dst_addrgen = InterleavedAddrGenFast<false>{
        .bank_base_address = tensor_address0,
        .page_size = tensor0_page_size,
        .data_format = get_dataformat(cb1_id),
    };

    uint32_t tile_id = 0;

    DPRINT << "OPENING CONNECTION\n";
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open_finish();
    }

    DPRINT << "Starting writer kernel\n";
    for (tile_id = 0; tile_id < num_tiles; tile_id++) {
        cb_reserve_back(cb0_id, 1);
        uint32_t l1_read_addr = get_write_ptr(cb0_id);
        uint64_t noc0_dest_noc_addr = get_noc_addr(0, 0, l1_read_addr);
        noc_async_read_tile(tile_id, tensor0_addrgen, l1_read_addr);

        l1_read_addr += data_size;
        noc_async_read_barrier();

        cb_push_back(cb0_id, 1);

        size_t l1_read_addr_t = static_cast<size_t>(l1_read_addr);

        size_t dst_l1_noc_addr = get_noc_addr(tile_id, dst_addrgen, 0 /*offset*/, 0 /*noc_id*/);
        DPRINT << "Writing to ETH\n";
        write_and_advance_local_read_address_for_fabric_write(
            dst_l1_noc_addr, pkt_hdr_forward, pkt_hdr_backward, fabric_connection, l1_read_addr_t, data_size);
        DPRINT << "Done Writing to ETH\n";
        DPRINT << out_ready_sem_noc0_x << ", " << out_ready_sem_noc0_y << "\n";
        DPRINT << semaphore << "\n";
    }

    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, semaphore, 0);
    // Write the mcast packet (forward)
    auto* pkt_hdr_fwd = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    pkt_hdr_fwd->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_noc_addr_in_pkt,
        static_cast<uint16_t>(1),  // increment 1
        32});

    fabric_connection.get_forward_connection().wait_for_empty_write_slot();
    pkt_hdr_fwd->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, 0});
    fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
        reinterpret_cast<uint32_t>(pkt_hdr_fwd), sizeof(PACKET_HEADER_TYPE));

    // while (*reinterpret_cast<volatile uint32_t*>(semaphore) != 1) {
    //     DPRINT << "Writing...... " << *reinterpret_cast<volatile uint32_t*>(semaphore) << "\n";
    // }
    DPRINT << "Closing connection\n";
    fabric_connection.close();
    DPRINT << "Done writer.\n";
    //
}
