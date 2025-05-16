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
#include <cmath>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;

constexpr uint32_t in_fwd_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t in_bwd_cb_index = get_compile_time_arg_val(1);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    DPRINT << "HELLO FROM READER\n";
    size_t arg_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_num_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tiles_per_col = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_per_buffer = get_arg_val<uint32_t>(arg_idx++);
    uint32_t device_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t device_order = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tensor_page_size = get_arg_val<uint32_t>(arg_idx++);
    uint32_t local_semaphore = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_semaphore);

    auto tensor0_addrgen = InterleavedAddrGenFast<true>{
        .bank_base_address = input_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(in_fwd_cb_index)};

    auto output_tensor_addrgen = InterleavedAddrGenFast<true>{
        .bank_base_address = output_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(in_fwd_cb_index)};

    uint32_t tile_id = 0;
    for (tile_id = 0; tile_id < input_num_tiles; tile_id++) {
        cb_reserve_back(in_fwd_cb_index, 1);
        auto tensor_tile_addr = tensor0_addrgen.get_noc_addr(tile_id);
        noc_async_read(tensor_tile_addr, get_write_ptr(in_fwd_cb_index), input_tensor_page_size);
        noc_async_read_barrier();
        cb_push_back(in_fwd_cb_index, 1);
    }

    uint32_t max_tiles_per_dst = num_tiles_per_buffer;

    uint32_t fwd_start_tile = 0;
    uint32_t fwd_iter_start_tile = 0;

    uint32_t fwd_end_tile = input_num_tiles / 2;
    uint32_t fwd_tiles_in_iter = std::min(fwd_end_tile - fwd_start_tile, max_tiles_per_dst);
    uint32_t fwd_iter_end_tile = fwd_iter_start_tile + fwd_tiles_in_iter;

    uint32_t bwd_start_tile = input_num_tiles / 2;
    uint32_t bwd_iter_start_tile = bwd_start_tile;
    uint32_t bwd_end_tile = input_num_tiles;
    uint32_t bwd_tiles_in_iter = std::min(bwd_end_tile - bwd_start_tile, max_tiles_per_dst);
    uint32_t bwd_iter_end_tile = bwd_iter_start_tile + bwd_tiles_in_iter;

    uint32_t iter_totals = std::ceil(input_num_tiles * 1.0 / max_tiles_per_dst);

    // DPRINT << "iter_totals: " << iter_totals << "\n";
    // DPRINT << "fwd_start_tile: " << fwd_start_tile << "\n";
    // DPRINT << "fwd_iter_start_tile: " << fwd_iter_start_tile << "\n";
    // DPRINT << "fwd_iter_end_tile: " << fwd_iter_end_tile << "\n";
    // DPRINT << "bwd_start_tile: " << bwd_start_tile << "\n";
    // DPRINT << "bwd_iter_start_tile: " << bwd_iter_start_tile << "\n";
    // DPRINT << "bwd_iter_end_tile: " << bwd_iter_end_tile << "\n";

    // For each device, read the output tensor from previous device into cb for forwarding
    for (int device_iter = 0; device_iter < 7; device_iter++) {
        uint32_t fwd_device_to_process = (((device_order - device_iter) % 8) + 8) % 8;
        uint32_t bwd_device_to_process = (((device_order + device_iter) % 8) + 8) % 8;

        fwd_iter_start_tile = 0;
        fwd_iter_end_tile = fwd_iter_start_tile + fwd_tiles_in_iter;
        bwd_iter_start_tile = input_num_tiles / 2;
        bwd_iter_end_tile = bwd_iter_start_tile + bwd_tiles_in_iter;

        DPRINT << "max_tiles_per_dst: " << max_tiles_per_dst << "\n";
        DPRINT << "iter_totals: " << iter_totals << "\n";
        DPRINT << "fwd_iter_start_tile: " << fwd_iter_start_tile << "\n";
        DPRINT << "fwd_iter_end_tile: " << fwd_iter_end_tile << "\n";
        DPRINT << "bwd_iter_start_tile: " << bwd_iter_start_tile << "\n";
        DPRINT << "bwd_iter_end_tile: " << bwd_iter_end_tile << "\n";

        for (uint32_t iter = 0; iter < iter_totals; iter++) {
            DPRINT << "Writing total of " << fwd_iter_end_tile - fwd_iter_start_tile << " tiles in fwd cb\n";
            for (tile_id = fwd_iter_start_tile; tile_id < fwd_iter_end_tile; tile_id++) {
                cb_reserve_back(in_fwd_cb_index, 1);
                // DPRINT << "READER FWD: " << device_order << " Get NOC addr for tile "
                //        << fwd_device_to_process * input_num_tiles + tile_id << "\t";
                // DPRINT << "Get NOC addr PARTS for tile " << fwd_device_to_process << " " << input_num_tiles << " "
                //        << tile_id << "\n";]
                uint32_t offset = (tile_id / tiles_per_col) * (8 * tiles_per_col);
                uint32_t target_tile_id = offset + fwd_device_to_process * tiles_per_col + tile_id % tiles_per_col;
                // DPRINT << "READER FWD: " << device_order << " maps " << fwd_device_to_process << " order " << tile_id
                // << " to " << target_tile_id << "\t" << " using offset " << offset << " and prev device offset " <<
                // fwd_device_to_process*tiles_per_col << " and mod " << tile_id % tiles_per_col << " and sum " <<
                // offset + fwd_device_to_process*tiles_per_col + tile_id % tiles_per_col << "\n";
                uint64_t tile_addr = output_tensor_addrgen.get_noc_addr(target_tile_id);
                noc_async_read(tile_addr, get_write_ptr(in_fwd_cb_index), input_tensor_page_size);
                noc_async_read_barrier();
                cb_push_back(in_fwd_cb_index, 1);
            }

            for (tile_id = bwd_iter_start_tile; tile_id < bwd_iter_end_tile; tile_id++) {
                cb_reserve_back(in_bwd_cb_index, 1);
                // DPRINT << "READER BWD: " << device_order << " Get NOC addr for tile "
                //    << bwd_device_to_process * input_num_tiles + tile_id << "\n";
                uint32_t offset = (tile_id / tiles_per_col) * (8 * tiles_per_col);
                uint32_t target_tile_id = offset + bwd_device_to_process * tiles_per_col + tile_id % tiles_per_col;
                // DPRINT << "READER BWD: " << device_order << " maps " << bwd_device_to_process << " order " << tile_id
                // << " to " << target_tile_id << "\t" << " using offset " << offset << " and prev device offset " <<
                // bwd_device_to_process*tiles_per_col << " and mod " << tile_id % tiles_per_col << " and sum " <<
                // offset + bwd_device_to_process*tiles_per_col + tile_id % tiles_per_col << "\n";
                uint64_t tile_addr = output_tensor_addrgen.get_noc_addr(target_tile_id);
                noc_async_read(tile_addr, get_write_ptr(in_bwd_cb_index), input_tensor_page_size);
                noc_async_read_barrier();
                cb_push_back(in_bwd_cb_index, 1);
            }
            // DPRINT << "READER: Waiting for semaphore value: " << (device_iter)*iter_totals + iter + 1
            //        << " that is in device iter " << device_iter << " and iter " << iter << " and got "
            //        << *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr_ptr) << "\n";

            while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr_ptr) !=
                   (uint32_t)(device_iter)*iter_totals + iter + 1);
            // DPRINT << "GOT SEMAPHORE VAL " << (device_iter)*iter_totals + iter + 1 << "\n";

            fwd_iter_start_tile = fwd_iter_end_tile;
            bwd_iter_start_tile = bwd_iter_end_tile;
            fwd_iter_end_tile = fwd_iter_start_tile + fwd_tiles_in_iter;
            bwd_iter_end_tile = bwd_iter_start_tile + bwd_tiles_in_iter;
        }
    }

    // DPRINT << "DONE READER\n";
}
