// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
/*
#include <stdint.h>

#include "dataflow_api.h"

FORCE_INLINE void fill_with_val(uint32_t begin_addr, uint32_t n, uint32_t val) {
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    for (uint32_t i = 0; i < n; ++i) {
        ptr[i] = val;
    }
}

void kernel_main() {

    constexpr uint32_t cb_id_in0 = 0;

    constexpr uint32_t num_padding_rows = get_compile_time_arg_val(3);
    const uint32_t total_num_rows = get_compile_time_arg_val(4);
    const uint32_t ncores = get_compile_time_arg_val(5);
    const uint32_t third_dim = get_compile_time_arg_val(6);
    const uint32_t tile_width = get_compile_time_arg_val(7);

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t unpadded_X_size = get_arg_val<uint32_t>(1);
    const uint32_t pad_value = get_arg_val<uint32_t>(2);
    const uint32_t core_number = get_arg_val<uint32_t>(3);

    DPRINT << "num_padding_rows: " << num_padding_rows << ENDL();
    DPRINT << "total_num_rows: " << total_num_rows << ENDL();
    DPRINT << "ncores: " << ncores << ENDL();
    DPRINT << "third_dim: " << third_dim << ENDL();
    DPRINT << "tile_width: " << tile_width << ENDL();

    DPRINT << "src_addr: " << src_addr << ENDL();
    DPRINT << "unpadded_X_size: " << unpadded_X_size << ENDL();
    DPRINT << "pad_value: " << pad_value << ENDL();
    DPRINT << "core_number: " << core_number << ENDL();




    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;

#define stick_size_is_pow2 get_compile_time_arg_val(1) == 1
#if (stick_size_is_pow2)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(2);
    const InterleavedPow2AddrGen<src0_is_dram> s = {
        .bank_base_address = src_addr, .log_base_2_of_page_size = log_base_2_of_page_size};
#else
    const InterleavedAddrGen<src0_is_dram> s = {.bank_base_address = src_addr, .page_size = unpadded_X_size};
#endif

    auto read_block = [&](uint32_t num_rows,
                          uint32_t mul,
                          uint32_t size_per_row_per_block,
                          uint32_t start_row_id,
                          uint32_t start_column_id,
                          uint32_t width_size,
                          uint32_t size_2d,
                          uint32_t element_size) {
        uint32_t onetile = 1;
        uint32_t padding_rows = num_rows == 32 ? 0 : 32 - num_rows;
        bool has_rows = (num_rows + padding_rows) > 0;

        cb_reserve_back(cb_id_in0, onetile * has_rows);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

        uint32_t original_addr = get_write_ptr(cb_id_in0);
        for (uint32_t k = start_row_id; k < start_row_id + num_rows; k++) {
            uint64_t src_noc_addr = get_noc_addr(size_2d + k, s);

            // Read from DRAM to tmp buffer
            DPRINT << "start writing at: " << l1_write_addr << ENDL();
            noc_async_read(src_noc_addr + start_column_id, l1_write_addr, width_size);

            // pad the row for the last core if needed
            uint32_t prev_size = start_column_id;
            uint32_t this_block_size = unpadded_X_size - prev_size;
            DPRINT << "prev_size: " << prev_size << ENDL();
            DPRINT << "this_block_size: " << this_block_size << ENDL();
            if (this_block_size < width_size) {
                uint32_t to_pad = width_size - this_block_size;
                DPRINT << "TO PAD: " << to_pad << ENDL();
                DPRINT << "START PADDING FROM: " <<  l1_write_addr + this_block_size << ENDL();
                fill_with_val(l1_write_addr + this_block_size + element_size, (to_pad) >> 2, pad_value);
            }
            //else if (prev_size > unpadded_X_size) {
            //    fill_with_val(l1_write_addr, (width_size) >> 2, pad_value);
            //}

            // Block before copying data from tmp to cb buffer
            noc_async_read_barrier();
            l1_write_addr += width_size;

            // pushing one tile at a time because the current LLK tilize implementation doesn't support tilizing more
            // than one tile per column at the same time this needs to be fixed
            //if (k > 0 && k % tile_width == 0) {
            //    cb_push_back(cb_id_in0, onetile * has_rows);
            //    cb_reserve_back(cb_id_in0, onetile * has_rows);
            //}
        }

        // pad in the height dim if needed
        fill_with_val(l1_write_addr, padding_rows * (width_size >> 2), pad_value);
        l1_write_addr += padding_rows * width_size;

        //auto* ptr_orig = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(original_addr);
        //for (uint32_t ii1 = 0; ii1 < 1024; ii1 = ii1+1) {
        //    if (ii1 % 16 == 0) {
        //     DPRINT << "CHECK HERE ";
        //    }
        //    DPRINT << "value at i1 = " << (uint32_t)ii1 <<  " is: " << BF16((uint16_t)ptr_orig[ii1]) << ENDL();
        //}

        cb_push_back(cb_id_in0, 1 * has_rows);
    };

    const uint32_t size_per_row_per_block = get_arg_val<uint32_t>(4);
    const uint32_t blocks_per_core = get_arg_val<uint32_t>(5);
    const uint32_t width_size = get_arg_val<uint32_t>(6);
    const uint32_t tile_height = get_arg_val<uint32_t>(7);   //move to compile time


    DPRINT << "size_per_row_per_block: " << size_per_row_per_block << ENDL();
    DPRINT << "blocks_per_core: " << blocks_per_core << ENDL();
    DPRINT << "width_size: " << width_size << ENDL();
    DPRINT << "tile_height: " << tile_height << ENDL();



    uint32_t size_2d = 0;
    uint32_t element_size = 2;
    for (uint32_t dim3 = 0; dim3 < third_dim; dim3++) {
        DPRINT << "OUTPUT FOR DIM :" << dim3 << ENDL();
        uint32_t start_row_id = get_arg_val<uint32_t>(8);
        uint32_t start_column_id = get_arg_val<uint32_t>(9);
        DPRINT << "start_row_id: " << start_row_id << ENDL();
        DPRINT << "start_column_id: " << start_column_id << ENDL();

        for (uint32_t b = 0; b < blocks_per_core; b++) {
            DPRINT <<" before read block " <<b << ENDL();
            uint32_t this_block_num_rows = tile_height;
            if (start_row_id + tile_height > total_num_rows) {
                this_block_num_rows = total_num_rows - start_row_id;
            }
            read_block(
                this_block_num_rows,
                core_number,
                size_per_row_per_block,
                // padded_size_per_row,
                start_row_id,
                start_column_id,
                width_size,
                size_2d,
                element_size);
            if (start_column_id + width_size < unpadded_X_size){
                start_column_id += width_size;
            }
            else{
                start_column_id =0;
                start_row_id += tile_height;
            }
            DPRINT << "start column id: " << start_column_id <<ENDL();
            DPRINT << "start row id: " << start_row_id <<ENDL();
            DPRINT <<" after read block " <<b << ENDL();
        }
        size_2d += total_num_rows;
    }

}


*/

// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

FORCE_INLINE void fill_with_val(uint32_t begin_addr, uint32_t n, uint32_t val) {
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    for (uint32_t i = 0; i < n; ++i) {
        ptr[i] = val;
    }
}

void kernel_main() {
    constexpr uint32_t cb_id_in0 = 0;

    constexpr uint32_t num_padding_rows = get_compile_time_arg_val(3);
    const uint32_t total_num_rows = get_compile_time_arg_val(4);
    const uint32_t ncores = get_compile_time_arg_val(5);
    const uint32_t third_dim = get_compile_time_arg_val(6);
    const uint32_t tile_width = get_compile_time_arg_val(7);
    const uint32_t element_size = get_compile_time_arg_val(8);

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t unpadded_X_size = get_arg_val<uint32_t>(1);
    const uint32_t pad_value = get_arg_val<uint32_t>(2);
    const uint32_t core_number = get_arg_val<uint32_t>(3);
    /*
    DPRINT << "num_padding_rows: " << num_padding_rows << ENDL();
    DPRINT << "total_num_rows: " << total_num_rows << ENDL();
    DPRINT << "ncores: " << ncores << ENDL();
    DPRINT << "third_dim: " << third_dim << ENDL();
    DPRINT << "tile_width: " << tile_width << ENDL();

    DPRINT << "src_addr: " << src_addr << ENDL();
    DPRINT << "unpadded_X_size: " << unpadded_X_size << ENDL();
    DPRINT << "pad_value: " << pad_value << ENDL();
    DPRINT << "core_number: " << core_number << ENDL();
    */

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;

#define stick_size_is_pow2 get_compile_time_arg_val(1) == 1
#if (stick_size_is_pow2)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(2);
    const InterleavedPow2AddrGen<src0_is_dram> s = {
        .bank_base_address = src_addr, .log_base_2_of_page_size = log_base_2_of_page_size};
#else
    const InterleavedAddrGen<src0_is_dram> s = {.bank_base_address = src_addr, .page_size = unpadded_X_size};
#endif

    auto read_block = [&](uint32_t num_rows,
                          uint32_t mul,
                          uint32_t size_per_row_per_block,
                          uint32_t start_row_id,
                          uint32_t start_column_id,
                          uint32_t width_size,
                          uint32_t size_2d,
                          uint32_t element_size,
                          uint32_t single_block_size) {
        uint32_t padding_rows = num_rows == 32 ? 0 : 32 - num_rows;
        bool has_rows = (num_rows + padding_rows) > 0;

        cb_reserve_back(cb_id_in0, single_block_size * has_rows);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

        uint32_t original_addr = get_write_ptr(cb_id_in0);
        for (uint32_t k = start_row_id; k < start_row_id + num_rows; k++) {
            uint64_t src_noc_addr = get_noc_addr(size_2d + k, s);

            // Read from DRAM to tmp buffer
            // DPRINT << "start writing at: " << l1_write_addr << ENDL();
            noc_async_read(src_noc_addr + start_column_id, l1_write_addr, width_size * single_block_size);

            // pad the row for the last core if needed
            uint32_t prev_size = start_column_id;
            uint32_t this_block_size = unpadded_X_size - prev_size;
            // DPRINT << "prev_size: " << prev_size << ENDL();
            // DPRINT << "this_block_size: " << this_block_size << ENDL();
            if (this_block_size < width_size * single_block_size) {
                uint32_t to_pad = width_size * single_block_size - this_block_size;
                // DPRINT << "TO PAD: " << to_pad << ENDL();
                // DPRINT << "START PADDING FROM: " <<  l1_write_addr + this_block_size << ENDL();
                fill_with_val(l1_write_addr + this_block_size + element_size, (to_pad) >> 2, 5);
            }
            // else if (prev_size > unpadded_X_size) {
            //     fill_with_val(l1_write_addr, (width_size) >> 2, pad_value);
            // }

            // Block before copying data from tmp to cb buffer
            noc_async_read_barrier();
            l1_write_addr += width_size * single_block_size;

            // pushing one tile at a time because the current LLK tilize implementation doesn't support tilizing more
            // than one tile per column at the same time this needs to be fixed
            // if (k > 0 && k % tile_width == 0) {
            //    cb_push_back(cb_id_in0, onetile * has_rows);
            //    cb_reserve_back(cb_id_in0, onetile * has_rows);
            //}
        }

        // pad in the height dim if needed
        // DPRINT << "padding rows at the end: " << padding_rows << " and num of rows initially: " << num_rows <<
        // ENDL(); DPRINT << "single_block_size: " << single_block_size << ENDL(); DPRINT << "width_size: " <<
        // width_size << ENDL(); DPRINT << "l1_write_addr: " << l1_write_addr << ENDL();

        // fill_with_val(l1_write_addr, padding_rows * (width_size * single_block_size >> 2), 3);
        // l1_write_addr += padding_rows * width_size * single_block_size;

        for (uint32_t pad_row = 0; pad_row < padding_rows; pad_row++) {
            fill_with_val(l1_write_addr, (width_size * single_block_size >> 2), 3);
            l1_write_addr += width_size * single_block_size;
        }

        // auto* ptr_orig = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(original_addr);
        // for (uint32_t ii1 = 0; ii1 < 2048; ii1 = ii1+1) {
        //     if (ii1 % 16 == 0) {
        //      DPRINT << "CHECK HERE ";
        //     }
        //     DPRINT << "value at i1 = " << (uint32_t)ii1 <<  " is: " << BF16((uint16_t)ptr_orig[ii1]) << ENDL();
        // }

        cb_push_back(cb_id_in0, single_block_size * has_rows);
    };

    const uint32_t size_per_row_per_block = get_arg_val<uint32_t>(4);
    const uint32_t blocks_per_core = get_arg_val<uint32_t>(5);
    const uint32_t width_size = get_arg_val<uint32_t>(6);
    const uint32_t tile_height = get_arg_val<uint32_t>(7);  // move to compile time

    // DPRINT << "size_per_row_per_block: " << size_per_row_per_block << ENDL();
    // DPRINT << "blocks_per_core: " << blocks_per_core << ENDL();
    // DPRINT << "width_size: " << width_size << ENDL();
    // DPRINT << "tile_height: " << tile_height << ENDL();

    uint32_t size_2d = 0;
    for (uint32_t dim3 = 0; dim3 < third_dim; dim3++) {
        // DPRINT << "OUTPUT FOR DIM :" << dim3 << ENDL();
        uint32_t start_row_id = get_arg_val<uint32_t>(8);
        uint32_t start_column_id = get_arg_val<uint32_t>(9);
        uint32_t single_block_size_row_arg = get_arg_val<uint32_t>(10);
        // DPRINT << "single_block_size_row_arg: " << single_block_size_row_arg << ENDL();
        uint32_t single_block_size_col_arg = get_arg_val<uint32_t>(11);
        // DPRINT << "single_block_size_col_arg: " << single_block_size_col_arg << ENDL();
        for (uint32_t b = 0; b < single_block_size_col_arg; b++) {
            // DPRINT <<" before read column " <<b << ENDL();
            // DPRINT << "start_row_id: " << start_row_id << ENDL();
            // DPRINT << "start_column_id: " << start_column_id << ENDL();
            uint32_t this_block_num_rows = tile_height;
            if (start_row_id + tile_height > total_num_rows) {
                this_block_num_rows = total_num_rows - start_row_id;
            }
            if (this_block_num_rows > 0) {
                read_block(
                    this_block_num_rows,
                    core_number,
                    size_per_row_per_block,
                    // padded_size_per_row,
                    start_row_id,
                    start_column_id,
                    width_size,
                    size_2d,
                    element_size,
                    single_block_size_row_arg);

                // if (start_column_id + width_size * single_block_size < unpadded_X_size){
                //     start_column_id += width_size * single_block_size;
                // }
                // else{
                //     start_column_id =0;
                //     start_row_id += tile_height;
                // }
                // DPRINT <<" after read block " <<b << ENDL();
            }
            start_row_id += tile_height;
        }
        size_2d += total_num_rows;
    }
}
