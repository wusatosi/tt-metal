// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr BufferType buffer0_type = static_cast<BufferType>(get_compile_time_arg_val(0));
constexpr uint32_t cb0_id = get_compile_time_arg_val(1);  // for receiver buffer id
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(2);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_devices = get_compile_time_arg_val(4);
constexpr uint32_t num_tiles = get_compile_time_arg_val(5);
constexpr uint32_t ring_index = get_compile_time_arg_val(6);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);                    // address of output tensor
    address_t receiver_base_address = get_arg_val<address_t>(arg_idx++);
    const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);          // address of global semaphore
    const uint32_t sync_semaphore = get_semaphore(get_arg_val<uint32_t>(arg_idx++));  // local semaphore
    uint32_t sender_noc_x = get_arg_val<uint32_t>(arg_idx++);                         // nox for sender core
    uint32_t sender_noc_y = get_arg_val<uint32_t>(arg_idx++);                         // noy for sender core
    volatile tt_l1_ptr uint32_t* sync_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_semaphore);
    *(sync_semaphore_ptr) = VALID;

    // print every compile and runtime arg in uint32_t
    DPRINT << "ct args: \n";
    DPRINT << "buffer0_type: " << (uint32_t)buffer0_type << "\n";
    DPRINT << "cb0_id: " << (uint32_t)cb0_id << "\n";
    DPRINT << "packet_size_in_pages: " << (uint32_t)packet_size_in_pages << "\n";
    DPRINT << "tensor0_page_size: " << (uint32_t)tensor0_page_size << "\n";
    DPRINT << "num_devices: " << (uint32_t)num_devices << "\n";
    DPRINT << "num_tiles: " << (uint32_t)num_tiles << "\n";
    DPRINT << "ring_index: " << (uint32_t)ring_index << "\n";

    DPRINT << "rt args: \n";
    DPRINT << "tensor_address0: " << (uint32_t)tensor_address0 << "\n";
    DPRINT << "out_ready_sem_bank_addr: " << (uint32_t)out_ready_sem_bank_addr << "\n";
    DPRINT << "sync_semaphore: " << (uint32_t)sync_semaphore << "\n";
    DPRINT << "sender_noc_x: " << (uint32_t)sender_noc_x << "\n";
    DPRINT << "sender_noc_y: " << (uint32_t)sender_noc_y << "\n";

    // interleaved addrgen
    constexpr bool is_dram = buffer0_type == tt::tt_metal::BufferType::DRAM;
    auto tensor0_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = tensor_address0, .page_size = tensor0_page_size, .data_format = get_dataformat(cb0_id)};

    // auto temp0_addrgen = InterleavedAddrGenFast<1>{
    //     .bank_base_address = receiver_base_address, .page_size = tensor0_page_size, .data_format =
    //     get_dataformat(cb0_id)};

    const InterleavedAddrGen<1> s0 = {.bank_base_address = receiver_base_address, .page_size = tensor0_page_size};
    for (uint32_t step = 0; step < 8; step++) {
        while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr) < (step + 1));
        DPRINT << "waitval done\n";
        uint32_t start_tile_id = ((ring_index - step + num_devices) % num_devices) * num_tiles;
        uint32_t tile_id_end = start_tile_id + num_tiles;
        uint32_t tile_id = start_tile_id;
        uint32_t count = 0;
        while (tile_id < tile_id_end) {
            DPRINT << "tile_id: " << tile_id << "\n";
            // cb_wait_front(cb0_id, packet_size_in_pages);
            // const uint32_t l1_read_address_base = get_read_ptr(cb0_id);
            // uint32_t l1_read_addr = l1_read_address_base;
            uint64_t l1_read_addr = get_noc_addr(count, s0);
            uint32_t num_pages_to_read = std::min(tile_id_end - tile_id, packet_size_in_pages);
            DPRINT << "num_pages_to_read: " << num_pages_to_read << "\n";
            for (uint32_t j = 0; j < num_pages_to_read; j++) {
                noc_async_write_tile(tile_id, tensor0_addrgen, l1_read_addr);
                DPRINT << "noc_async_write_tile done\n";
                l1_read_addr += tensor0_page_size;
                tile_id++;
            }
            count++;

            noc_async_read_barrier();
            DPRINT << "noc_async_read_barrier done\n";
            cb_push_back(cb0_id, packet_size_in_pages);
            DPRINT << "cb_push_back done\n";
        }

        // set sender's semaphore
        uint64_t sender_sem_addr = get_noc_addr(sender_noc_x, sender_noc_y, sync_semaphore);
        DPRINT << "sender_sem_addr: " << sender_sem_addr << "\n";
        noc_semaphore_set_remote(sync_semaphore, sender_sem_addr);
        DPRINT << "noc_semaphore_set_remote done\n";
    }

    DPRINT << "DONE \n";
}
