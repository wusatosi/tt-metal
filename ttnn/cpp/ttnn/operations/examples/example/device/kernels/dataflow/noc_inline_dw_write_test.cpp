// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "dprint.h"

void dprint_be_binary(uint32_t be32) {
    DPRINT << "byte enable mask : ";
    for (int i = 31; i >= 0; i--) {
        DPRINT << ((be32 & (1 << i)) ? "1" : "0");
    }
    DPRINT << "\n";
}

void check_dram_status(uint64_t noc_addr, uint8_t cb_id) {
    cb_reserve_back(cb_id, 1);
    auto l1_write_addr = get_write_ptr(cb_id);
    noc_async_read(noc_addr, l1_write_addr, 64, noc_index);  // read one row in face0 (4-byte)
    noc_async_read_barrier();
    cb_push_back(cb_id, 1);

    cb_wait_front(cb_id, 1);
    // DPRINT only first row of tile
    for (int32_t r = 0; r < 1; ++r) {
        SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT_DATA0({
            DPRINT << (uint)r << " --READ--cin1-- "
                   << TileSlice(cb_id, 0, sr, TSLICE_INPUT_CB, TSLICE_RD_PTR, true, false) << ENDL();
        });
    }
    DPRINT << "\n";
    cb_pop_front(cb_id, 1);
}

template <uint8_t noc_mode = DM_DEDICATED_NOC>
inline __attribute__((always_inline)) void noc_fast_write_dw_inline_moreh(
    uint32_t noc,
    uint32_t cmd_buf,
    uint32_t val,
    uint64_t dest_addr,
    uint32_t be32,
    uint32_t static_vc,
    bool mcast,
    bool posted = false) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        if (posted) {
            inc_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc, 1);
        } else {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, 1);
        }
    }
    bool static_vc_alloc = true;
    uint32_t noc_cmd_field = (static_vc_alloc ? NOC_CMD_VC_STATIC : 0x0) | NOC_CMD_STATIC_VC(static_vc) | NOC_CMD_CPY |
                             NOC_CMD_WR | NOC_CMD_WR_INLINE | NOC_CMD_WR_BE |  // add WR_BE flag.
                             (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0) |
                             (posted ? 0x0 : NOC_CMD_RESP_MARKED);
    dprint_be_binary(be32);

    while (!noc_cmd_buf_ready(noc, cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_DATA, val);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, dest_addr & 0xFFFFFFFF);
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, be32);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        if (posted) {
            noc_posted_writes_num_issued[noc] += 1;
        } else {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += 1;
        }
    }
}

void kernel_main() {
    DPRINT << "----- Kernel Start ----" << ENDL();

    uint32_t noc_x = get_arg_val<uint32_t>(0);
    uint32_t noc_y = get_arg_val<uint32_t>(1);
    uint32_t dst_addr = get_arg_val<uint32_t>(2);

    constexpr uint8_t cb_print = 0;
    constexpr uint32_t first_tile = 0;
    constexpr uint32_t one_tile_size_1B = 4096;

    InterleavedAddrGenFast<true> dst_addr_gen = {
        .bank_base_address = dst_addr, .page_size = one_tile_size_1B, .data_format = get_dataformat(cb_print)};

    // noc address given by address generator
    uint64_t noc_addr1 = get_noc_addr(0, dst_addr_gen);
    DPRINT << HEX();
    DPRINT << "noc_addr1 - 0x" << noc_addr1 << ENDL();
    // noc address with physical noc_xy of worker cores on (0, 1)
    uint64_t noc_addr2 = get_noc_addr(noc_x, noc_y, dst_addr, noc_index);
    DPRINT << "noc_addr2 - 0x" << noc_addr2 << ENDL();

    // value to write
    uint32_t val = 0x0BADF00D;

    // 1. noc_inline_dw_write with noc_addr1
    DPRINT << "Case 1 - noc_inline_dw_write on noc address to dram" << ENDL();
    DPRINT << "Before-write : " << ENDL();
    check_dram_status(noc_addr1, cb_print);

    noc_inline_dw_write(noc_addr1, val, 0xF, noc_index);
    noc_async_write_barrier(noc_index);

    DPRINT << "After-write : " << ENDL();
    check_dram_status(noc_addr1, cb_print);

    // 2. noc_inline_dw_write with noc_addr2
    DPRINT << "Case 2 - noc_inline_dw_write on noc address to worker core (SRAM)" << ENDL();
    DPRINT << "Before-write print with noc_addr2 : " << ENDL();
    check_dram_status(noc_addr2, cb_print);

    noc_inline_dw_write(noc_addr2, val, 0xF, noc_index);
    noc_async_write_barrier(noc_index);

    DPRINT << "After-write print with noc_addr2: " << ENDL();
    check_dram_status(noc_addr2, cb_print);

    // 3. test with custom noc_inline_dw_write defined in this file.
    DPRINT << "Case 3 - custom noc_inline_dw_write on noc address to dram with full be mask" << ENDL();
    DPRINT << "Before-write : " << ENDL();
    check_dram_status(noc_addr1, cb_print);

    uint32_t be32 = 0xFFFFFFFF;  // 32 bytes enabled
    noc_fast_write_dw_inline_moreh(
        noc_index,
        write_at_cmd_buf,
        val,
        noc_addr1,
        be32,
        NOC_UNICAST_WRITE_VC,
        /*mcast*/ false,
        /*posted*/ false);
    noc_async_write_barrier(noc_index);

    DPRINT << "After-write : " << ENDL();
    check_dram_status(noc_addr1, cb_print);

    DPRINT << "----- Kernel End ----" << ENDL();
}
