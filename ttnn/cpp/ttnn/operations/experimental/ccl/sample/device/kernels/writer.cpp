// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;

constexpr uint32_t cb1_id = get_compile_time_arg_val(0);
constexpr uint32_t data_size = get_compile_time_arg_val(1);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    uint32_t num_tiles = get_arg_val<uint32_t>(arg_idx++);
    const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);

    DPRINT << "rt args: \n";
    DPRINT << "tensor_address0: " << (uint32_t)tensor_address0 << "\n";
    DPRINT << "num_tiles: " << (uint32_t)num_tiles << "\n";

    // interleaved addrgen
    constexpr bool is_dram = true;
    uint32_t tensor0_page_size = 1088;
    auto tensor0_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = tensor_address0, .page_size = tensor0_page_size, .data_format = get_dataformat(cb1_id)};

    DPRINT << "tensor -> CB: " << (uint32_t)cb1_id << "\n";
    DPRINT << "data size: " << (uint32_t)data_size << "\n";

    uint32_t tile_id = 0;

    // cb_wait_front(cb1_id, 1);
    // DPRINT << "!!!!!! got cb_wait_front\n";
    // uint32_t l1_write_addr = get_read_ptr(cb1_id);
    // DPRINT << "got read ptr: " << (uint32_t)l1_write_addr << "\n";
    // noc_async_write_tile(tile_id, tensor0_addrgen, l1_write_addr);

    // for (tile_id = 0; tile_id < num_tiles; tile_id++) {
    //     while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr) < 1);

    //     DPRINT << "tile_id: " << tile_id << "\t" << "num_tiles: " << (uint32_t)num_tiles << "\n";
    //     cb_wait_front(cb1_id, 1);
    //     DPRINT << "!!!!!! got cb_wait_front\n";
    //     uint32_t l1_write_addr = get_read_ptr(cb1_id);
    //     DPRINT << "got read ptr: " << (uint32_t)l1_write_addr << "\n";
    //     noc_async_write_tile(tile_id, tensor0_addrgen, l1_write_addr);
    //     DPRINT << "got noc_async_write\n";
    //     l1_write_addr += data_size;
    //     noc_async_write_barrier();
    //     DPRINT << "got noc_async_write_barrier\n";
    //     cb_pop_front(cb1_id, 1);
    //     DPRINT << "got cb_pop_front\n";
    // }

    DPRINT << "DONE \n";
}
