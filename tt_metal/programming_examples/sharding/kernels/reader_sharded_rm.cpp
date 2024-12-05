// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "debug/dprint.h"
#include <stdint.h>
#include "dataflow_api.h"

#include "debug/ring_buffer.h"

// export TT_METAL_DPRINT_CORES='(0,0)-(0,3)' in order to see DPRINT messages
constexpr int WALL_CLOCK_HIGH_INDEX = 1;
constexpr int WALL_CLOCK_LOW_INDEX = 0;

uint64_t get_time() {
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    while (true) {
        uint32_t high = p_reg[WALL_CLOCK_HIGH_INDEX];
        uint32_t low = p_reg[WALL_CLOCK_LOW_INDEX];
        if (p_reg[WALL_CLOCK_HIGH_INDEX] == high) {
            return (static_cast<uint64_t>(high) << 32) | low;
        }
    }
}

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t stick_size = get_arg_val<uint32_t>(1);
    const uint32_t shard_height = get_arg_val<uint32_t>(2);
    const uint32_t shard_width_bytes = get_arg_val<uint32_t>(3);
    const uint32_t padded_offset_bytes = get_arg_val<uint32_t>(4);
    const uint32_t start_id = get_arg_val<uint32_t>(5);
    const uint32_t current_core = get_arg_val<uint32_t>(6);

    const uint32_t do_multicasts = get_arg_val<uint32_t>(7);
    const uint32_t l1_buffer_address = get_arg_val<uint32_t>(8);
    const uint32_t mcast_address = get_arg_val<uint32_t>(9);
    const uint32_t num_mcast_dests = get_arg_val<uint32_t>(10);

    uint64_t start_time = get_time();

    uint64_t one_second = 1024 * 1024 * 1024;
    uint64_t timeout = 100 * one_second;

    if (do_multicasts) {
        uint32_t count = 0;
        uint32_t noc = noc_index;

#define USE_CUSTOM_WRITE
#ifdef USE_CUSTOM_WRITE
        bool linked = false;
        bool multicast_path_reserve = true;

        uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                 NOC_CMD_STATIC_VC(NOC_MULTICAST_WRITE_VC) | (linked ? NOC_CMD_VC_LINKED : 0x0) |
                                 ((multicast_path_reserve ? NOC_CMD_PATH_RESERVE : 0) | NOC_CMD_BRCST_PACKET) |
                                 NOC_CMD_RESP_MARKED;

        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CTRL, noc_cmd_field);
        NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, mcast_address);
#endif

        while (get_time() - start_time < timeout) {
            uint64_t destination_address = l1_buffer_address + (rand() % 4096);
            destination_address = destination_address & ~0xf;
            uint32_t size = 1 + (rand() % 4095);
#ifdef USE_CUSTOM_WRITE
            WAYPOINT("NMPW");
            DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc, dst_noc_addr_multicast, src_local_l1_addr, size);
            while (!noc_cmd_buf_ready(noc, write_cmd_buf));
            WAYPOINT("NWPD");

            NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_TARG_ADDR_LO, l1_buffer_address);
            NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_LO, (uint32_t)destination_address);
            NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_AT_LEN_BE, size);
            NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += num_mcast_dests;
#else
            noc_async_write_multicast_one_packet(
                l1_buffer_address,
                get_noc_addr_helper(mcast_address, destination_address),
                1 + (rand() % 4095),
                num_mcast_dests);
#endif
            WATCHER_RING_BUFFER_PUSH(++count);

#if 0
            uint32_t count2 = 0;
            bool sent = 0;
            WAYPOINT("AAAW");
            while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) {
                if (++count2 == 1000000 && !sent) {
                    sent = true;
                    WATCHER_RING_BUFFER_PUSH(NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED));
                    WATCHER_RING_BUFFER_PUSH(noc_nonposted_writes_acked[noc_index]);
                }
            }
            WAYPOINT("AAAD");
#endif
        }

        uint32_t count2 = 0;
        bool sent = 0;
        WAYPOINT("BAAW");
        while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) {
            if (++count2 == 1000000 && !sent) {
                sent = true;
                WATCHER_RING_BUFFER_PUSH(NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED));
                WATCHER_RING_BUFFER_PUSH(noc_nonposted_writes_acked[noc_index]);
            }
        }

        WAYPOINT("BAAD");
        // noc_async_write_barrier();

    } else {
        constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
        constexpr bool src_is_dram = get_compile_time_arg_val(1) == 1;
        const InterleavedAddrGen<src_is_dram> s0 = {.bank_base_address = src_addr, .page_size = stick_size};
        cb_reserve_back(cb_id_in0, shard_height);
        while (get_time() - start_time < timeout) {
            uint32_t stick_id = start_id;
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
            DPRINT_DATA0(DPRINT << "Core (0," << current_core << "): ");
            for (uint32_t h = 0; h < shard_height; ++h) {
                uint64_t src_noc_addr = get_noc_addr(stick_id, s0);
                noc_async_read(src_noc_addr, l1_write_addr, stick_size);
                // print both BFloat16 values that are packed into the page
                uint32_t* read_ptr = (uint32_t*)l1_write_addr;
                DPRINT_DATA0(DPRINT << (uint16_t)*read_ptr << " ");
                DPRINT_DATA0(DPRINT << (uint16_t)(*read_ptr >> 16) << " ");
                stick_id++;
                l1_write_addr += padded_offset_bytes;
            }
            DPRINT_DATA0(DPRINT << ENDL());
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, shard_height);
    }
}
