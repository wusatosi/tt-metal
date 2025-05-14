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

#define UNICAST_HDR \
    tt::tt_fabric::MulticastRoutingCommandHeader { 1, static_cast<uint8_t>(1) }

#define INC_HDR(addr, flush) \
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader { addr, static_cast<uint16_t>(1), 32, flush }

using address_t = uint32_t;
using tt::tt_metal::BufferType;

constexpr uint32_t in_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t dst_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t header_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t is_forward = get_compile_time_arg_val(3);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t global_sem_addr_sent = get_arg_val<uint32_t>(arg_idx++);
    uint32_t global_sem_addr_can_receive = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_noc_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t device_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t device_order = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_num_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tensor_page_size = get_arg_val<uint32_t>(arg_idx++);
    uint32_t local_semaphore = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    volatile tt_l1_ptr uint32_t* local_sem_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_semaphore);

    volatile tt_l1_ptr uint32_t* global_sem_addr_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_sem_addr_sent);

    volatile tt_l1_ptr uint32_t* global_sem_addr_can_receive_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_sem_addr_can_receive);

    uint64_t local_noc_sem_addr = safe_get_noc_addr(out_noc_x, out_noc_y, local_semaphore, 0);

    uint64_t out_global_sem_sent_noc_addr = safe_get_noc_addr(out_noc_x, out_noc_y, global_sem_addr_sent, 0);

    uint64_t out_global_sem_can_receive_noc_addr =
        safe_get_noc_addr(out_noc_x, out_noc_y, global_sem_addr_can_receive, 0);

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
    volatile PACKET_HEADER_TYPE* pkt_hdr_fwd =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_bwd =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);

    pkt_hdr_fwd->to_chip_multicast(UNICAST_HDR);
    pkt_hdr_bwd->to_chip_multicast(UNICAST_HDR);

    auto in_tensor_addrgen = InterleavedAddrGenFast<true>{
        .bank_base_address = input_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(in_cb_index)};

    auto out_tensor_addrgen = InterleavedAddrGenFast<true>{
        .bank_base_address = output_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(in_cb_index)};

    // Copy all tiles from cb to output tensor (local data)
    for (uint32_t i = 0; i < input_num_tiles; i++) {
        cb_wait_front(in_cb_index, 1);
        uint64_t dest_addr = out_tensor_addrgen.get_noc_addr(device_order * input_num_tiles + i);
        noc_async_write(get_read_ptr(in_cb_index), dest_addr, input_tensor_page_size);
        noc_async_write_barrier();
        cb_pop_front(in_cb_index, 1);
    }

    fabric_connection.open_finish();

    auto conn = is_forward ? fabric_connection.get_forward_connection() : fabric_connection.get_backward_connection();
    auto bwd_conn =
        is_forward ? fabric_connection.get_backward_connection() : fabric_connection.get_forward_connection();

    int device_direction = is_forward ? -1 : 1;

    constexpr uint32_t start_tile = 0;
    constexpr uint32_t end_tile = 2;

    for (uint32_t device_iter = 0; device_iter < 7; device_iter++) {
        // Sync up, let next device know we can receive data
        pkt_hdr_bwd->to_noc_unicast_atomic_inc(INC_HDR(out_global_sem_can_receive_noc_addr, true));
        bwd_conn.wait_for_empty_write_slot();
        bwd_conn.send_payload_flush_non_blocking_from_address(
            reinterpret_cast<uint32_t>(pkt_hdr_bwd), sizeof(PACKET_HEADER_TYPE));

        // Wait until the next device told us it can recieve data
        while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_sem_addr_can_receive_ptr) < (device_iter + 1));

        uint32_t device_to_process = (((device_order + device_iter * device_direction) % 8) + 8) % 8;
        uint32_t prev_device = (((device_to_process + device_direction) % 8) + 8) % 8;

        uint32_t l1_dst_addr = get_write_ptr(dst_cb_index);
        DPRINT << "DST CB L1 ADDR: " << l1_dst_addr << "\n";

        // Send all eth packets from CB
        for (uint32_t i = start_tile; i < end_tile; i++) {
            cb_wait_front(in_cb_index, 1);
            uint32_t l1_read_addr = get_read_ptr(in_cb_index);
            uint64_t dst_noc_addr = safe_get_noc_addr(out_noc_x, out_noc_y, l1_dst_addr, 0);

            pkt_hdr_fwd->to_noc_unicast_write(
                tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr}, input_tensor_page_size);
            conn.wait_for_empty_write_slot();
            conn.send_payload_without_header_non_blocking_from_address(l1_read_addr, input_tensor_page_size);
            conn.send_payload_flush_non_blocking_from_address((uint32_t)pkt_hdr_fwd, sizeof(PACKET_HEADER_TYPE));
            noc_async_writes_flushed();

            l1_dst_addr += input_tensor_page_size;

            cb_pop_front(in_cb_index, 1);
        }

        // Increase global semaphore of next device, let next device know it can read data
        DPRINT << "Sending semaphore increase...\n";
        pkt_hdr_fwd->to_noc_unicast_atomic_inc(INC_HDR(out_global_sem_sent_noc_addr, true));
        conn.wait_for_empty_write_slot();
        conn.send_payload_flush_non_blocking_from_address(
            reinterpret_cast<uint32_t>(pkt_hdr_fwd), sizeof(PACKET_HEADER_TYPE));

        DPRINT << "WRITER: Waiting on semaphore val: " << (device_iter + 1) << " and got "
               << *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_sem_addr_sent_ptr) << "\n";
        // Wait to recieve the data
        while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_sem_addr_sent_ptr) < (device_iter + 1));

        DPRINT << "!!! GOT SEMAPHORE !!!\n";

        // Recieved from eth -> output tensor
        uint32_t read_ptr = get_write_ptr(dst_cb_index);
        for (uint32_t i = start_tile; i < end_tile; i++) {
            uint64_t dest_addr = out_tensor_addrgen.get_noc_addr(prev_device * input_num_tiles + i);
            noc_async_write(read_ptr, dest_addr, input_tensor_page_size);
            noc_async_write_barrier();
            read_ptr += input_tensor_page_size;
        }
        DPRINT << "Increasing local semaphore...\n";

        noc_semaphore_inc(local_noc_sem_addr, 1);
        noc_async_writes_flushed();
    }

    fabric_connection.close();
    DPRINT << "Done writer.\n";
}
