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

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);  // address of input tensor
    const uint32_t sync_semaphore = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    volatile tt_l1_ptr uint32_t* sync_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_semaphore);

    // print every compile and runtime arg in uint32_t
    DPRINT << "ct args: \n";
    DPRINT << "buffer0_type: " << (uint32_t)buffer0_type << "\n";
    DPRINT << "cb0_id: " << (uint32_t)cb0_id << "\n";
    DPRINT << "packet_size_in_pages: " << (uint32_t)packet_size_in_pages << "\n";
    DPRINT << "tensor0_page_size: " << (uint32_t)tensor0_page_size << "\n";

    DPRINT << "rt args: \n";
    DPRINT << "tensor_address0: " << (uint32_t)tensor_address0 << "\n";

    // interleaved addrgen
    constexpr bool is_dram = buffer0_type == tt::tt_metal::BufferType::DRAM;
    auto tensor0_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = tensor_address0, .page_size = tensor0_page_size, .data_format = get_dataformat(cb0_id)};

    // step 0: tle_id = ring_index * num_tiles_perd_device

    for (uint32_t step = 0; step < 8; step++) {
        uint32_t tile_id = ((ring_index - step + num_devices) % num_devices) * num_tiles;
        uint32_t tile_id_end = start_tile_id + num_tiles;
        while (tile_id < start_tile_id + num_tiles) {
            DPRINT << "tile_id: " << tile_id << "\n";
            cb_reserve_back(cb0_id, packet_size_in_pages);
            const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
            uint32_t l1_write_addr = l1_write_addr_base;

            uint32_t num_pages_to_read = std::min(tile_id_end - tile_id, packet_size_in_pages);
            for (uint32_t j = 0; j < num_pages_to_read; j++) {
                noc_async_read_tile(tile_id, tensor0_addrgen, l1_write_addr);
                l1_write_addr += tensor0_page_size;
                tile_id++;
            }

            noc_async_read_barrier();
            cb_push_back(cb0_id, packet_size_in_pages);
        }

        // wait for receiver writer semaphore
        // set it to zero
        noc_semaphore_wait(sync_semaphore_ptr, 1);
        noc_semaphore_set(sync_semaphore_ptr, 0);
    }

    DPRINT << "DONE \n";
}
