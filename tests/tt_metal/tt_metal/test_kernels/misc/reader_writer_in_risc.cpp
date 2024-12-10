// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

void kernel_main() {
    uint32_t src_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_dram_addr = get_arg_val<uint32_t>(1);
    uint32_t src_bank_id = get_arg_val<uint32_t>(2);
    uint32_t dst_bank_id = get_arg_val<uint32_t>(3);
    uint32_t add_value = get_arg_val<uint32_t>(4);
    uint32_t tile_size_x = get_arg_val<uint32_t>(5);
    uint32_t tile_size_y = get_arg_val<uint32_t>(6);
    uint32_t scaling_sem_idx = get_arg_val<uint32_t>(7);
    uint32_t tile_toggle_val = get_arg_val<uint32_t>(8);

    // NoC coords (x,y) depending on DRAM location on-chip
    uint64_t src_dram_noc_addr = get_noc_addr_from_bank_id<true>(src_bank_id, src_dram_addr);
    uint64_t dst_dram_noc_addr = get_noc_addr_from_bank_id<true>(dst_bank_id, dst_dram_addr);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;  // index=0

    // single-tile ublocks
    uint32_t tile_size_bytes = get_tile_size(cb_id_in0);

    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

    // Read data from DRAM -> L1 circular buffers
    noc_async_read(src_dram_noc_addr, l1_write_addr, tile_size_bytes);
    noc_async_read_barrier();

    // Write data from L1 circulr buffer (in0) -> DRAM
    uint16_t* data = (uint16_t*)l1_write_addr;
    uint32_t* sem_addr = reinterpret_cast<uint32_t*>(get_semaphore(scaling_sem_idx));
    for (uint32_t i = 0; i < tile_size_x * tile_size_y; i++) {
        if (i % (tile_size_x * tile_toggle_val) == 0 and i) {
            noc_semaphore_inc(get_noc_addr(get_semaphore(scaling_sem_idx)), 2);
            noc_async_atomic_barrier();
        }
        data[i] = (*sem_addr) * add_value + data[i];
    }
    noc_async_write(l1_write_addr, dst_dram_noc_addr, tile_size_bytes);
    noc_async_write_barrier();
}
