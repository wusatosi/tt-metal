// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include <stdint.h>
#include "dataflow_api.h"

#define MASK_64      0xFFFFFFC0
#define OFFSET_64    0x0000003F
#define MASK_16      0xFFFFFFF0
#define OFFSET_16    0x0000000F

//Note for the future:
//I am assuming the size of a tile is at least 64 bytes for DDR or 16 bytes for L1 and at the tile dimensions are multiples of 2 (e.g, 2x4, 6x8 ...)
//Note when using 32x32 tiles the above are true even if our data type is 1 bit which is still unsupported. Only an issue in tiny tiles
template <bool guaranteed_16B_alligned, bool read_async>
FORCE_INLINE
void tt_memmove (
    const uint32_t dst_l1_addr,
    const uint32_t src_l1_addr,
    const uint32_t bytes)
{
    //Uses noc_async_read when possible to copy the data over
    if constexpr (guaranteed_16B_alligned)
    {
        noc_async_read(src_l1_addr, get_noc_add(source_buffer), bytes);
        if constexpr (!read_async) {noc_async_read_barrier();}
    }
    else
    {
        if ((dst_l1_addr&OFFSET_16) == (src_l1_addr&OFFSET_16))
        {
            noc_async_read(src_l1_addr, get_noc_add(source_buffer), bytes);
            if constexpr (!read_async) {noc_async_read_barrier();}
        }
        else
        {
            memmove(dst_l1_addr, src_l1_addr, bytes);
        }
    }
}


template <bool is_l1, bool reads_alligned_64, bool reads_alligned_16>
FORCE_INLINE
void unaligned_noc_async_read (
    const uint32_t src_noc_addr,
    const uint32_t target_buffer_addr,
    const uint32_t scratch_buffer_addr,
    const uint32_t bytes_to_read
)
{
    //Perform an unaligned noc async read
    //size of the scratch buffer must be 64 bytes more than bytes_to_read
    if constexpr (reads_alligned_64 || (is_l1 && reads_alligned_16))
    {
        noc_async_read(src_noc_addr, target_buffer_addr, bytes_to_read);
        return;
    }
    else if constexpr (is_l1)
    {
        if ((src_noc_addr & OFFSET_16)==0 && (target_buffer_addr & OFFSET_16)==0)
        {
            noc_async_read(src_noc_addr, target_buffer_addr, bytes_to_read);
            return;
        }
        noc_async_read(src_noc_addr&MASK_16, scratch_buffer_addr,bytes_to_read+16);
        noc_async_read_barrier();
        tt_memmove<false,false>(target_buffer_addr, scratch_buffer_addr + src_noc_addr&OFFSET_16, bytes_to_read);
        return;
    }
    else
    {
        if (((src_noc_addr & OFFSET_64)==0 && (target_buffer_addr & OFFSET_64)==0))
        {
            noc_async_read(src_noc_addr, target_buffer_addr, bytes_to_read);
            return;
        }
        noc_async_read(src_noc_addr&MASK_64, scratch_buffer_addr,bytes_to_read+64);
        noc_async_read_barrier();
        tt_memmove<false,false>(target_buffer_addr, scratch_buffer_addr + src_noc_addr&OFFSET_64, bytes_to_read);
        return;
    }
}

void kernel_main() {
    //We are guranteed to be in 2D going to 2D

    const uint32_t src_tensor               = get_arg_val<uint32_t>(0);
    const uint32_t dst_tensor               = get_arg_val<uint32_t>(1);
    const uint32_t cb_id_in0                = get_arg_val<uint32_t>(2);
    const uint32_t src_w_byte               = get_arg_val<uint32_t>(3);
    const uint32_t src_h                    = get_arg_val<uint32_t>(4);
    const uint32_t src_d                    = get_arg_val<uint32_t>(5);
    const uint32_t dst_w_byte               = get_arg_val<uint32_t>(6);
    const uint32_t dst_h                    = get_arg_val<uint32_t>(7);
    const uint32_t dst_d                    = get_arg_val<uint32_t>(8);
    const uint32_t src_h_tiles              = get_arg_val<uint32_t>(10);
    const uint32_t dst_w_tiles              = get_arg_val<uint32_t>(12);
    const uint32_t dst_h_tiles              = get_arg_val<uint32_t>(13);
    const uint32_t rows_to_pad              = get_arg_val<uint32_t>(14);
    const uint32_t pad_high                 = get_arg_val<uint32_t>(15);
    const uint32_t pad_low                  = get_arg_val<uint32_t>(16);

    //Compile time arguments
    constexpr bool src0_is_dram                     = (get_compile_time_arg_val(0) == 1);
    constexpr bool tensor_is_l1                     = (get_compile_time_arg_val(0) == 0);
    constexpr uint32_t datum_size                   = get_compile_time_arg_val<uint32_t>(1);
    constexpr uint32_t tile_width                   = get_compile_time_arg_val<uint32_t>(2);
    constexpr uint32_t tile_height                  = get_compile_time_arg_val<uint32_t>(3);
    constexpr uint32_t src_w_tiles                  = get_compile_time_arg_val<uint32_t>(4);

    //Constant expressions

    constexpr uint32_t face_width_bytes = datum_size * tile_width / 2;//By assumption this is an integer
    constexpr uint32_t face_height = tile_height / 2; //by assumption this is an integer
    constexpr bool     face_alligned_16 = (face_width_bytes%16==0);
    constexpr bool     face_alligned_64 = (face_width_bytes%64==0);
    constexpr uint32_t face_size_bytes = face_height * face_width_bytes;
    constexpr uint32_t tile_size_bytes  = tile_width * tile_height * datum_size; //By assumption this is a multiple of 64


    //Precompute some values
    const uint32_t bytes_in_right_tile = src_w_byte % (face_width_bytes * 2);
    const uint32_t pad_bytes_in_right_tile = (face_width_bytes * 2) - bytes_in_right_tile;
    const bool thin_right_tile = bytes_in_right_tile <= face_width_bytes;
    const bool has_right_pad = bytes_in_right_tile != 0;
    const uint32_t row_pad_size = rows_to_pad * (face_width_bytes * 2 * src_w_tiles);
    constexpr uint32_t side_pad_size = tile_width * datum_size;
    //Actual readsize + one face to overread. We make it 64 aligned and add 64 in case the alignment shrunk the size
    const uint32_t min_line_size = ((src_w_tiles) * (face_width_bytes) * 2 +face_width_bytes)&MASK_64 + 64;
    //Get CB
    cb_reserve_back(cb_id_in0, 1);
    const uint32_t buffer_base = get_write_ptr(cb_id_in0);

    //Subdivide CB into our scratchpad buffers
    const uint32_t buffer_start = buffer_base&MASK_64+64;
    const uint32_t tiling_buffer          = buffer_start; //Alligned buffer enough to hold dst_w_tiles
    const uint32_t padding_buffer         = tiling_buffer + (dst_w_tiles*tile_size_bytes)&MASK_64+64; //Alligned buffer enough to hold one line (min_line_size)
    const uint32_t line_buffer            = padding_buffer + min_line_size; //Alligned buffer enough to hold one line (min_line_size)
    const uint32_t async_scratch_buffer   = line_buffer + min_line_size; //scratch_buffer needs to be face_width_bytes +64 bytes in size as per unaligned_noc_async_read requirements

    const InterleavedAddrGen<src0_is_dram> s = {
        .bank_base_address = src_tensor,
        .page_size = tile_size_bytes
    };

    const InterleavedAddrGen<src0_is_dram> d = {
        .bank_base_address = dst_tensor,
        .page_size = tile_size_bytes
    };
    //Prepare_the_pad_buffer
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(padding_buffer);
    ptr[0] = pad_high;
    ptr[1] = pad_low;
    ptr[2] = pad_high;
    ptr[3] = pad_low;
    num_written = 4;
    while (num_written < min_line_size)
    {
        //Each time double the amount in the pad buffer until it is filled
        uint32_t num_to_write = (2*num_written > min_line_size) ? min_line_size-num_written : num_written;
        tt_memmove<true,false>(padding_buffer+num_written,padding_buffer,num_to_write);
        num_to_write = num_to_write * 2;
    }

    //Do the work
    uint32_t dst_left_tile = 0;
    uint32_t dst_tile_col = 0;
    uint32_t dst_tile_row = 0;
    bool dst_face_left = true;
    bool dst_face_down = true;
    uint32_t dst_index = 0;
    uint64_t dest_noc_addr;

    uint64_t row_noc_addresses[src_w_tiles];//Holds the NOC addresses of all the tiles in this source row
    for (int depth=0; depth<src_d; depth++)
    {
        uint32_t src_left_tile = depth * src_h_tiles * src_w_tiles;
        uint32_t rows_to_add = dst_h;
        uint32_t this_row_offset = 0;
        bool bottom_face = true;
        while (rows_to_add > 0)
        {
            //Process the height dimension
            if (bottom_face)
            {
                //This is a new tile to process
                for (int col=0; col < src_w_tiles; col++)
                {
                    row_noc_addresses[col] = get_noc_addr(col + src_left_tile, s);
                }
                src_left_tile = src_left_tile + src_w_tiles;
            }
            uint32_t row_offset = (bottom_face) ? 0: face_size_bytes * 2;//Offset within each tile of the left face to copy
            uint32_t rows_to_copy = face_height < rows_to_add ? face_height: rows_to_add;
            for (uint32_t row = 0; row < face_height; row ++)
            {
                //copy over a row
                uint32_t line_cur_pointer = line_buffer;
                uint32_t writing_source;
                if (row < rows_to_copy)
                {
                    //We are copying in a real row of data
                    writing_source = line_buffer
                    //We are copying in a real row of data
                    for (int col=0; col < src_w_tiles-1; col++)
                    {
                        //copy over the row from one tile ignoring the right most tile
                        unaligned_noc_async_read<tensor_is_l1, face_alligned_64, face_alligned_16>(
                            row_noc_addresses[col]+row_offset,
                            line_cur_pointer,
                            async_scratch_buffer,
                            face_width_bytes);
                        unaligned_noc_async_read<tensor_is_l1, face_alligned_64, face_alligned_16>(
                            row_noc_addresses[col]+row_offset + face_size_bytes,
                            line_cur_pointer + face_width_bytes,
                            async_scratch_buffer,
                            face_width_bytes);
                        line_cur_pointer = line_cur_pointer + (face_width_bytes*2);
                    }
                    //Copy the right_most tile
                    if (thin_right_tile)
                    {
                        unaligned_noc_async_read<tensor_is_l1, face_alligned_64, face_alligned_16>(
                            row_noc_addresses[col]+row_offset,
                            line_cur_pointer,
                            async_scratch_buffer,
                            bytes_in_right_tile);
                    }
                    else
                    {
                        unaligned_noc_async_read<tensor_is_l1, face_alligned_64, face_alligned_16>(
                            row_noc_addresses[col]+row_offset,
                            line_cur_pointer,
                            async_scratch_buffer,
                            face_width_bytes);
                        unaligned_noc_async_read<tensor_is_l1, face_alligned_64, face_alligned_16>(
                            row_noc_addresses[col]+row_offset + face_size_bytes,
                            line_cur_pointer + face_width_bytes,
                            async_scratch_buffer,
                            bytes_in_right_tile-face_width_bytes);
                    }
                    line_cur_pointer = line_cur_pointer + bytes_in_right_tile;
                    if (has_right_pad)
                    {
                        tt_memmove<false,true>(line_cur_pointer, padding_buffer, pad_bytes_in_right_tile);
                        line_cur_pointer = line_cur_pointer + pad_bytes_in_right_tile;
                    }
                    noc_async_read_barrier();
                }
                else
                {
                    //We will copy in the line from the padding buffer instead
                    writing_source = padding_buffer;
                }
                //write a tile line from writing_source to the dest structure
                uint32_t readable = (src_w_tiles) * ((face_width_bytes) * 2);
                bool should_add_write_barrier = false;
                while (readable > 0)
                {
                    //iterate until we have read in all the data from the line buffer into tiling buffer copying tiles out as we finish with them
                    //Place a barrier after changing to a new row of tiles (inside loop) or if we issued any noc_async_writes (by setting should_add_write_barrier to true)

                    //Go to the start of the tiling buffer
                    write_buffer_address = tiling_buffer;
                    //Go to the correct tile
                    uint32_t current_tile = write_buffer_address+dst_tile_col * tile_size_bytes;
                    //Go to the correct face
                    write_buffer_address = dst_face_down ? current_tile : current_tile + face_size_bytes * 2;
                    write_buffer_address = dst_face_left ? write_buffer_address : write_buffer_address + face_size_bytes;
                    //Go to the correct row
                    write_buffer_address = write_buffer_address + dst_tile_row * face_width_bytes;
                    //Go to the correct index within the row
                    write_buffer_address = write_buffer_address + dst_index;
                    if (readable < face_width_bytes - dst_index)
                    {
                        tt_memmove<false,false>(write_buffer_address,writing_source, readable);
                        dst_index = dst_index + readable;
                        readable = 0;
                    }
                    else
                    {
                        tt_memmove<false,false>(write_buffer_address,writing_source, (face_width_bytes - dst_index));
                        dst_index = 0;
                        if (dst_face_left)
                        {
                            //Flip from left face to right face
                            dst_face_left = false;
                        }
                        else if (dst_tile_col != dst_w_tiles - 1)
                        {
                            if ((!dst_face_down) && dst_tile_row == face_height - 1)
                            {
                                //TODO Write the current tile out to dst_left_tile + dst_tile_col page
                                should_add_write_barrier = true;
                                dest_noc_addr = get_noc_addr(src_left_tile + dst_tile_col, d);
                                noc_async_write(current_tile,dest_noc_addr, dest_page_size_bytes);
                            }
                            //Go from right face to left face of next column
                            dst_tile_col = dst_tile_col + 1;
                            dst_face_left = true;
                        }
                        else if (dst_tile_row < face_height - 1)
                        {
                            //Go to the next row
                            dst_tile_row = dst_tile_row + 1;
                            dst_face_left = true;
                            dst_tile_col = 0;
                        }
                        else if (dst_face_down)
                        {
                            //Go to the upper faces
                            dst_face_down = false;
                            dst_tile_row = 0;
                            dst_face_left = true;
                            dst_tile_col = 0;
                        }
                        else
                        {
                            //The very last tile in the datastructure
                            //TODO Write this tile out to dst_left_tile + dst_w_tiles - 1 page and call write barrier
                            dest_noc_addr = get_noc_addr(src_left_tile + dst_tile_col, d);
                            noc_async_write(current_tile,dest_noc_addr, dest_page_size_bytes);
                            dst_face_down = true;
                            should_add_write_barrier = false;
                            dst_tile_row = 0;
                            dst_face_left = true;
                            dst_tile_col = 0;
                            dst_left_tile = dst_left_tile + dst_w_tiles;
                            noc_async_write_barrier();
                        }

                        readable = readable - (face_width_bytes - dst_index);
                    }
                }
                if (should_add_write_barrier)
                {
                    noc_async_write_barrier();
                }
                row_offset = row_offset + face_width_bytes;
            }
            rows_to_add = rows_to_add - rows_to_copy;
            bottom_face = !bottom_face;
        }
    }
}
