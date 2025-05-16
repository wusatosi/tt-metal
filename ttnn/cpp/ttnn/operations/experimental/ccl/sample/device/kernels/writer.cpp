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
#include <cmath>
#include <utility>

#define UNICAST_HDR \
    tt::tt_fabric::MulticastRoutingCommandHeader { 1, static_cast<uint8_t>(1) }

#define INC_HDR(addr, flush) \
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader { addr, static_cast<uint16_t>(1), 32, flush }

using address_t = uint32_t;
using tt::tt_metal::BufferType;

constexpr uint32_t in_fwd_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t in_bwd_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t dst_fwd_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t dst_bwd_cb_index = get_compile_time_arg_val(3);
constexpr uint32_t header_cb_index = get_compile_time_arg_val(4);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    // DPRINT << "Hello from writer.\n";

    size_t arg_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t global_sem_addr_sent = get_arg_val<uint32_t>(arg_idx++);
    uint32_t global_sem_addr_can_receive = get_arg_val<uint32_t>(arg_idx++);
    uint32_t noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t noc_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t device_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t device_order = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_num_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tiles_per_col = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_per_buffer = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tensor_page_size = get_arg_val<uint32_t>(arg_idx++);
    uint32_t local_semaphore = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    DPRINT << "input_num_tiles: " << input_num_tiles << ", tiles_per_row: " << tiles_per_row
           << ", tiles_per_col: " << tiles_per_col << "\n";

    volatile tt_l1_ptr uint32_t* local_sem_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_semaphore);

    volatile tt_l1_ptr uint32_t* global_sem_addr_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_sem_addr_sent);

    volatile tt_l1_ptr uint32_t* global_sem_addr_can_receive_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_sem_addr_can_receive);

    uint64_t local_noc_sem_addr = safe_get_noc_addr(noc_x, noc_y, local_semaphore, 0);

    uint64_t out_global_sem_sent_noc_addr = safe_get_noc_addr(noc_x, noc_y, global_sem_addr_sent, 0);

    uint64_t out_global_sem_can_receive_noc_addr = safe_get_noc_addr(noc_x, noc_y, global_sem_addr_can_receive, 0);

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
        .data_format = get_dataformat(in_fwd_cb_index)};

    auto out_tensor_addrgen = InterleavedAddrGenFast<true>{
        .bank_base_address = output_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(in_fwd_cb_index)};

    uint32_t max_tiles_per_dst = num_tiles_per_buffer;

    uint32_t iter_totals = std::ceil(input_num_tiles * 1.0 / max_tiles_per_dst);

    // Copy all tiles from cb to output tensor (local data)
    for (uint32_t i = 0; i < input_num_tiles; i++) {
        cb_wait_front(in_fwd_cb_index, 1);
        uint32_t offset = (i / tiles_per_col) * (8 * tiles_per_col);
        uint32_t target_tile_id = offset + device_order * tiles_per_col + i % tiles_per_col;
        uint64_t dest_addr = out_tensor_addrgen.get_noc_addr(target_tile_id);
        noc_async_write(get_read_ptr(in_fwd_cb_index), dest_addr, input_tensor_page_size);
        noc_async_write_barrier();
        cb_pop_front(in_fwd_cb_index, 1);
    }

    // DPRINT << "Opening FABRIC connection\n";
    fabric_connection.open_finish();
    // DPRINT << "FABRIC connection opened\n";

    auto fwd_conn = fabric_connection.get_forward_connection();
    auto bwd_conn = fabric_connection.get_backward_connection();

    for (uint32_t device_iter = 0; device_iter < 7; device_iter++) {
        uint32_t fwd_device_to_process = (((device_order - device_iter) % 8) + 8) % 8;
        uint32_t fwd_prev_device = (((fwd_device_to_process - 1) % 8) + 8) % 8;
        uint32_t bwd_device_to_process = (((device_order + device_iter) % 8) + 8) % 8;
        uint32_t bwd_prev_device = (((bwd_device_to_process + 1) % 8) + 8) % 8;

        uint32_t fwd_start_tile = 0;
        uint32_t fwd_end_tile = input_num_tiles / 2;
        uint32_t fwd_tiles_in_iter = std::min(fwd_end_tile - fwd_start_tile, max_tiles_per_dst);
        uint32_t fwd_iter_start_tile = fwd_start_tile;
        uint32_t fwd_iter_end_tile = fwd_iter_start_tile + fwd_tiles_in_iter;

        uint32_t bwd_start_tile = input_num_tiles / 2;
        uint32_t bwd_end_tile = input_num_tiles;
        uint32_t bwd_tiles_in_iter = std::min(bwd_end_tile - bwd_start_tile, max_tiles_per_dst);
        uint32_t bwd_iter_start_tile = bwd_start_tile;
        uint32_t bwd_iter_end_tile = bwd_iter_start_tile + bwd_tiles_in_iter;

        // show all important variables
        DPRINT << "max_tiles_per_dst: " << max_tiles_per_dst << "\n";
        DPRINT << "iter_totals: " << iter_totals << "\n";
        DPRINT << "fwd_iter_start_tile: " << fwd_iter_start_tile << "\n";
        DPRINT << "fwd_iter_end_tile: " << fwd_iter_end_tile << "\n";
        DPRINT << "bwd_iter_start_tile: " << bwd_iter_start_tile << "\n";
        DPRINT << "bwd_iter_end_tile: " << bwd_iter_end_tile << "\n";

        // DPRINT << "ITER TOTALS " << iter_totals << " and tiles in iter " << tiles_in_iter << "\n";
        for (uint32_t iter = 0; iter < iter_totals; iter++) {
            // DPRINT << "SYNC UP " << device_iter << "\n";
            //////// Let other devices know we can receive data and wait we can send
            // Sync up backward direction
            pkt_hdr_bwd->to_chip_multicast(UNICAST_HDR);
            pkt_hdr_bwd->to_noc_unicast_atomic_inc(INC_HDR(out_global_sem_can_receive_noc_addr, true));
            bwd_conn.wait_for_empty_write_slot();
            bwd_conn.send_payload_flush_non_blocking_from_address(
                reinterpret_cast<uint32_t>(pkt_hdr_bwd), sizeof(PACKET_HEADER_TYPE));

            // Sync up forward direction
            pkt_hdr_fwd->to_chip_multicast(UNICAST_HDR);
            pkt_hdr_fwd->to_noc_unicast_atomic_inc(INC_HDR(out_global_sem_can_receive_noc_addr, true));
            fwd_conn.wait_for_empty_write_slot();
            fwd_conn.send_payload_flush_non_blocking_from_address(
                reinterpret_cast<uint32_t>(pkt_hdr_fwd), sizeof(PACKET_HEADER_TYPE));

            // Wait until the both devices told us they can recieve data
            // DPRINT << "Stuck on semaphore\n";
            while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_sem_addr_can_receive_ptr) !=
                   2 * ((device_iter)*iter_totals + iter + 1));

            //////// Sending the data

            DPRINT << "Sending data\n";

            // Send to forward direction
            uint32_t l1_fwd_dst_addr = get_write_ptr(dst_fwd_cb_index);
            DPRINT << "DST CB L1 ADDR: " << l1_fwd_dst_addr << "\n";

            // Send all eth packets from CB
            DPRINT << "Expecting total of " << fwd_iter_end_tile - fwd_iter_start_tile
                   << " tiles in forward direction\n";
            for (uint32_t i = fwd_iter_start_tile; i < fwd_iter_end_tile; i++) {
                cb_wait_front(in_fwd_cb_index, 1);
                uint32_t l1_read_addr = get_read_ptr(in_fwd_cb_index);
                uint64_t dst_noc_addr = safe_get_noc_addr(noc_x, noc_y, l1_fwd_dst_addr, 0);

                pkt_hdr_fwd->to_chip_multicast(UNICAST_HDR);
                pkt_hdr_fwd->to_noc_unicast_write(
                    tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr}, input_tensor_page_size);
                fwd_conn.wait_for_empty_write_slot();
                fwd_conn.send_payload_without_header_non_blocking_from_address(l1_read_addr, input_tensor_page_size);
                fwd_conn.send_payload_flush_non_blocking_from_address(
                    (uint32_t)pkt_hdr_fwd, sizeof(PACKET_HEADER_TYPE));

                l1_fwd_dst_addr += input_tensor_page_size;

                cb_pop_front(in_fwd_cb_index, 1);
            }

            DPRINT << "Sending data to backward direction\n";

            // Increase global semaphore of next device, let next device know it can read data
            pkt_hdr_fwd->to_chip_multicast(UNICAST_HDR);
            pkt_hdr_fwd->to_noc_unicast_atomic_inc(INC_HDR(out_global_sem_sent_noc_addr, true));
            fwd_conn.wait_for_empty_write_slot();
            fwd_conn.send_payload_flush_non_blocking_from_address(
                reinterpret_cast<uint32_t>(pkt_hdr_fwd), sizeof(PACKET_HEADER_TYPE));

            // Send to backward direction
            uint32_t l1_bwd_dst_addr = get_write_ptr(dst_bwd_cb_index);

            // Send all eth packets from CB
            for (uint32_t i = bwd_iter_start_tile; i < bwd_iter_end_tile; i++) {
                cb_wait_front(in_bwd_cb_index, 1);
                // DPRINT << "Got backward cb " << i << "\n";
                uint32_t l1_read_addr = get_read_ptr(in_bwd_cb_index);
                uint64_t dst_noc_addr = safe_get_noc_addr(noc_x, noc_y, l1_bwd_dst_addr, 0);

                pkt_hdr_bwd->to_chip_multicast(UNICAST_HDR);
                pkt_hdr_bwd->to_noc_unicast_write(
                    tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr}, input_tensor_page_size);
                bwd_conn.wait_for_empty_write_slot();
                bwd_conn.send_payload_without_header_non_blocking_from_address(l1_read_addr, input_tensor_page_size);
                bwd_conn.send_payload_flush_non_blocking_from_address(
                    (uint32_t)pkt_hdr_bwd, sizeof(PACKET_HEADER_TYPE));

                l1_bwd_dst_addr += input_tensor_page_size;

                cb_pop_front(in_bwd_cb_index, 1);
            }

            DPRINT << "Sending data to forward direction\n";

            // Increase global semaphore of next device, let next device know it can read data
            pkt_hdr_bwd->to_chip_multicast(UNICAST_HDR);
            pkt_hdr_bwd->to_noc_unicast_atomic_inc(INC_HDR(out_global_sem_sent_noc_addr, true));
            bwd_conn.wait_for_empty_write_slot();
            bwd_conn.send_payload_flush_non_blocking_from_address(
                reinterpret_cast<uint32_t>(pkt_hdr_bwd), sizeof(PACKET_HEADER_TYPE));

            noc_async_writes_flushed();

            //////// Recieving data
            DPRINT << "Recieving data\n";

            // Wait to recieve the data
            while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_sem_addr_sent_ptr) !=
                   2 * ((device_iter)*iter_totals + iter + 1));

            DPRINT << "!!! GOT SEMAPHORE !!!\n";

            // Recieved fwd direction
            uint32_t read_ptr = get_write_ptr(dst_fwd_cb_index);
            // DPRINT << "WRITER: FWD ITER START " << fwd_iter_start_tile << " and END " << fwd_iter_end_tile << "\n";
            for (uint32_t i = fwd_iter_start_tile; i < fwd_iter_end_tile; i++) {
                uint32_t offset = (i / tiles_per_col) * (8 * tiles_per_col);
                uint32_t target_tile_id = offset + fwd_prev_device * tiles_per_col + i % tiles_per_col;
                // DPRINT << "WRITER FWD: " << device_order << " maps " << fwd_prev_device << " order " << i << " to "
                // << target_tile_id << "\t" << " using offset " << offset << " and prev device offset " <<
                // fwd_prev_device*tiles_per_col << " and mod " << i % tiles_per_col << " and sum " << offset +
                // fwd_prev_device*tiles_per_col + i % tiles_per_col << "\n";

                uint64_t dest_addr = out_tensor_addrgen.get_noc_addr(target_tile_id);
                noc_async_write(read_ptr, dest_addr, input_tensor_page_size);
                read_ptr += input_tensor_page_size;
            }

            // Recieved bwd direction
            read_ptr = get_write_ptr(dst_bwd_cb_index);
            // DPRINT << "WRITER: BWD ITER START " << bwd_iter_start_tile << " and END " << bwd_iter_end_tile << "\n";
            for (uint32_t i = bwd_iter_start_tile; i < bwd_iter_end_tile; i++) {
                uint32_t offset = (i / tiles_per_col) * (8 * tiles_per_col);
                uint32_t target_tile_id = offset + tiles_per_col * bwd_prev_device + i % tiles_per_col;
                // DPRINT << "WRITER BWD: " << device_order << " maps " << bwd_prev_device << " order " << i << " to "
                // << target_tile_id << "\t" << " using offset " << offset << " and prev device offset " <<
                // bwd_prev_device*tiles_per_col << " and mod " << i % tiles_per_col << "\n";
                uint64_t dest_addr = out_tensor_addrgen.get_noc_addr(target_tile_id);
                noc_async_write(read_ptr, dest_addr, input_tensor_page_size);
                read_ptr += input_tensor_page_size;
            }

            noc_async_write_barrier();
            noc_async_read_barrier();

            // DPRINT << "Increasing local semaphore on device iter " << device_iter << " and iter " << iter << "\n";

            noc_semaphore_inc(local_noc_sem_addr, 1);

            fwd_iter_start_tile = fwd_iter_end_tile;
            bwd_iter_start_tile = bwd_iter_end_tile;
            fwd_iter_end_tile = fwd_iter_start_tile + fwd_tiles_in_iter;
            bwd_iter_end_tile = bwd_iter_start_tile + bwd_tiles_in_iter;
        }
    }

    // final semaphor value
    // DPRINT << "WRITER: FINAL semaphore val: "
    //    << *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_sem_addr_sent_ptr) << "\n";

    fabric_connection.close();
    // DPRINT << "Done writer.\n";
}
