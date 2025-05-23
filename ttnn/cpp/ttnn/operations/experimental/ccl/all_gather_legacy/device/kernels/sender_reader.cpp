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

constexpr BufferType buffer0_type = static_cast<BufferType>(get_compile_time_arg_val(0));  // buffer type
constexpr uint32_t cb0_id = get_compile_time_arg_val(1);                                   // source buffer id
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(2);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_devices = get_compile_time_arg_val(4);
constexpr uint32_t num_tiles = get_compile_time_arg_val(5);  // num tiles per device
constexpr uint32_t ring_index = get_compile_time_arg_val(6);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);  // address of input tensor
    address_t tensor_address1 = get_arg_val<address_t>(arg_idx++);  // address of output tensor
    const uint32_t sync_semaphore = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    volatile tt_l1_ptr uint32_t* sync_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_semaphore);

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
    DPRINT << "sync_semaphore: " << (uint32_t)sync_semaphore << "\n";

    // interleaved addrgen
    constexpr bool is_dram = buffer0_type == tt::tt_metal::BufferType::DRAM;
    DPRINT << "is_dram: " << (uint32_t)is_dram << "\n";
    auto tensor0_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = tensor_address0, .page_size = tensor0_page_size, .data_format = get_dataformat(cb0_id)};

    auto tensor1_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = tensor_address1, .page_size = tensor0_page_size, .data_format = get_dataformat(cb0_id)};
    // step 0: tle_id = ring_index * num_tiles_perd_device

    for (uint32_t step = 0; step < 8; step++) {
        auto tensor_addrgen = step == 0 ? tensor0_addrgen : tensor1_addrgen;
        DPRINT << "step: " << step << "\n";
        uint32_t start_tile_id = ((ring_index - step + num_devices) % num_devices) * num_tiles;
        DPRINT << "start_tile_id: " << start_tile_id << "\n";
        uint32_t tile_id_end = step == 0 ? num_tiles : start_tile_id + num_tiles;
        DPRINT << "tile_id_end: " << tile_id_end << "\n";
        uint32_t tile_id = step == 0 ? 0 : start_tile_id;
        while (tile_id < tile_id_end) {
            DPRINT << "tile_id: " << tile_id << "\n";
            cb_reserve_back(cb0_id, packet_size_in_pages);
            const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
            uint32_t l1_write_addr = l1_write_addr_base;
            DPRINT << "l1_write_addr_base: " << l1_write_addr_base << "\n";

            uint32_t num_pages_to_read = std::min(tile_id_end - tile_id, packet_size_in_pages);
            DPRINT << "num_pages_to_read: " << num_pages_to_read << "\n";
            for (uint32_t j = 0; j < num_pages_to_read; j++) {
                noc_async_read_tile(tile_id, tensor_addrgen, l1_write_addr);
                noc_async_read_barrier();
                volatile tt_l1_ptr uint16_t* dst_noc2 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr);
                for (uint16_t value = 0; value < 2; value++) {
                    DPRINT << "value at " << (uint16_t)value << " is: " << BF16((uint16_t)dst_noc2[value]) << ENDL();
                }

                l1_write_addr += tensor0_page_size;
                tile_id++;
            }

            cb_push_back(cb0_id, packet_size_in_pages);
        }
        DPRINT << "after while loop\n";

        // wait for receiver writer semaphore
        // set it to zero
        noc_semaphore_wait(sync_semaphore_ptr, 1);
        noc_semaphore_set(sync_semaphore_ptr, 0);
        DPRINT << "after waiting for semaphore\n";
    }

    DPRINT << "DONE \n";
}
