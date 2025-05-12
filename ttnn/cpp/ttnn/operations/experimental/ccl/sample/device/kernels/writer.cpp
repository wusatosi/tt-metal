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

constexpr uint32_t cb_index = get_compile_time_arg_val(0);
constexpr uint32_t dst_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t header_cb_index = get_compile_time_arg_val(2);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t device_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t device_order = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_num_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tensor_page_size = get_arg_val<uint32_t>(arg_idx++);

    auto fabric_connection =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);

    // packet header cb
    cb_reserve_back(header_cb_index, 1);
    auto packet_header_buffer_addr_forward = get_write_ptr(header_cb_index);
    cb_push_back(header_cb_index, 1);
    cb_reserve_back(header_cb_index, 1);
    auto packet_header_buffer_addr_backward = get_write_ptr(header_cb_index);
    cb_push_back(header_cb_index, 1);

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
    pkt_hdr_forward->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(1)});
    pkt_hdr_backward->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(1)});

    auto in_tensor_addrgen = InterleavedAddrGenFast<true>{
        .bank_base_address = input_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_index)};

    auto out_tensor_addrgen = InterleavedAddrGenFast<true>{
        .bank_base_address = output_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_index)};

    // stage 0: cb -> output tensor
    DPRINT << "Device id: " << device_id << ", device order: " << device_order << "\n";
    DPRINT << "Reader reading...\n";
    DPRINT << "Num tiles: " << input_num_tiles << "\n";
    for (uint32_t i = 0; i < input_num_tiles; i++) {
        DPRINT << "STAGE 0: WAITING ON TILE: " << i << "\n";
        cb_wait_front(cb_index, 1);
        DPRINT << "STAGE 0: Tile: " << i << ", page size: " << input_tensor_page_size << "\n";
        uint64_t dest_addr = out_tensor_addrgen.get_noc_addr(device_order * input_num_tiles + i);
        DPRINT << "STAGE 0: For device order: " << device_order
               << ", writing tile: " << device_order * input_num_tiles + i << " to " << dest_addr << "\n";
        noc_async_write(get_read_ptr(cb_index), dest_addr, input_tensor_page_size);
        noc_async_write_barrier();
        cb_pop_front(cb_index, 1);
    }

    DPRINT << "Real done stage 0...\n";
    fabric_connection.open_finish();

    DPRINT << "Done stage 0...\n";
    // // stage 1: output tensor -> eth
    uint32_t prev_addr = 0;
    uint32_t l1_dst_addr = get_write_ptr(dst_cb_index);
    for (uint32_t i = 0; i < input_num_tiles; i++) {
        DPRINT << "STAGE 1: WAITING ON TILE: " << i << "\n";
        cb_wait_front(cb_index, 1);
        DPRINT << "STAGE 1: Tile: " << i << ", page size: " << input_tensor_page_size << "\n";
        // uint64_t dest_addr = out_tensor_addrgen.get_noc_addr(device_order*input_num_tiles + i);
        DPRINT << "STAGE 1: For device order: " << device_order
               << ", writing tile: " << device_order * input_num_tiles + i << "\n";
        uint32_t l1_read_addr = get_read_ptr(cb_index);
        if (prev_addr == 0) {
            prev_addr = l1_read_addr;
        } else {
            DPRINT << "ADDR JUMP " << prev_addr << " -> " << l1_read_addr << "\n";
            DPRINT << "ADDR DIFF " << l1_read_addr - prev_addr << "\n";
        }

        uint64_t dst_noc_addr = safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, l1_dst_addr, 0);

        pkt_hdr_forward->to_noc_unicast_write(
            tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr}, input_tensor_page_size);
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, input_tensor_page_size);
        fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)pkt_hdr_forward, sizeof(PACKET_HEADER_TYPE));
        noc_async_writes_flushed();

        l1_dst_addr += input_tensor_page_size;

        cb_pop_front(cb_index, 1);
    }
    DPRINT << "Done stage 1...\n";

    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);

    DPRINT << "Sending forward semaphore increase...\n";
    auto* pkt_hdr_fwd = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    pkt_hdr_fwd->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_noc_addr_in_pkt,
        static_cast<uint16_t>(1),  // increment 1
        32});

    pkt_hdr_fwd->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(1)});
    fabric_connection.get_forward_connection().wait_for_empty_write_slot();
    fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
        reinterpret_cast<uint32_t>(pkt_hdr_fwd), sizeof(PACKET_HEADER_TYPE));

    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr);

    noc_semaphore_wait(signal_semaphore_addr_ptr, 1);

    noc_async_read_barrier();
    noc_async_write_barrier();

    DPRINT << "!!! GOT SEMAPHORE !!!\n";

    DPRINT << "Stage 2\n";

    uint32_t read_ptr = get_read_ptr(dst_cb_index);
    for (uint32_t i = 0; i < input_num_tiles; i++) {
        auto prev_device = (((device_order - 1) % 8) + 8) % 8;
        DPRINT << "STAGE 2: WRITING TILE ID: " << prev_device * input_num_tiles + i << " on device " << prev_device
               << " for device " << device_order << " id " << device_id << "\n";
        uint64_t dest_addr = out_tensor_addrgen.get_noc_addr(prev_device * input_num_tiles + i);
        DPRINT << "STAGE 2: " << "tile id " << prev_device * input_num_tiles + i << " read ptr " << read_ptr
               << " dest addr " << dest_addr << "\n";
        noc_async_write(read_ptr, dest_addr, input_tensor_page_size);
        noc_async_write_barrier();
        read_ptr += input_tensor_page_size;
    }

    fabric_connection.close();
    DPRINT << "Done writer.\n";
}
