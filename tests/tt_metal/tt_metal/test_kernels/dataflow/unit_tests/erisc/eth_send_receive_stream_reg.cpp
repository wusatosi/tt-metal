// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/dprint.h"
#include "ethernet/dataflow_api.h"

FORCE_INLINE void eth_setup_handshake(uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
    }
}

void kernel_main() {
    bool is_sender = get_arg_val<uint32_t>(0) == 1;
    uint32_t handshake_addr = get_arg_val<uint32_t>(1);
    uint32_t stream_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t DEFAULT_ETH_TXQ = 0;

    uint32_t value = 0x1;

    // volatile tt_l1_ptr uint32_t* result_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_addr);

    // init stream reg
    NOC_STREAM_WRITE_REG(stream_id, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, 0);

    eth_setup_handshake(handshake_addr, is_sender);

    if (is_sender) {
        uint32_t src_addr = 0x00022100;
        uint32_t dest_addr = 0x00033300;
        uint32_t payload_size_bytes = 4128;

        uint32_t tx_cnt = ETH_READ_REG(0xFFB90030);
        uint32_t local_rn = ETH_READ_REG(0xFFB94040);
        uint32_t remote_rn = ETH_READ_REG(0xFFB94044);
        uint32_t dropped = ETH_READ_REG(0xFFB9404C);

        DPRINT << "tx_cnt: " << tx_cnt << " local_rn: " << local_rn << ", remote_rn: " << remote_rn
               << ", dropped: " << dropped << ENDL();

        while (internal_::eth_txq_is_busy(DEFAULT_ETH_TXQ)) {
        };
        internal_::eth_send_packet_bytes_unsafe(DEFAULT_ETH_TXQ, src_addr, dest_addr, payload_size_bytes);

        for (int i = 0; i < 4096; i++) {
            asm("nop");
        }

        uint32_t tx_cnt1 = ETH_READ_REG(0xFFB90030);
        uint32_t local_rn1 = ETH_READ_REG(0xFFB94040);
        uint32_t remote_rn1 = ETH_READ_REG(0xFFB94044);
        uint32_t dropped1 = ETH_READ_REG(0xFFB9404C);

        DPRINT << "tx_cnt: " << tx_cnt1 << " local_rn: " << local_rn1 << ", remote_rn: " << remote_rn1
               << ", dropped: " << dropped1 << ENDL();

        while (internal_::eth_txq_is_busy(DEFAULT_ETH_TXQ)) {
        };
        uint32_t addr = STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
        DPRINT << "stream reg addr " << HEX() << addr << ENDL();
        internal_::eth_write_remote_reg_no_txq_check(DEFAULT_ETH_TXQ, addr, value << REMOTE_DEST_BUF_WORDS_FREE_INC);

        while (internal_::eth_txq_is_busy(DEFAULT_ETH_TXQ)) {
        };

        for (int i = 0; i < 4096; i++) {
            asm("nop");
        }

        uint32_t tx_cnt2 = ETH_READ_REG(0xFFB90030);
        uint32_t local_rn2 = ETH_READ_REG(0xFFB94040);
        uint32_t remote_rn2 = ETH_READ_REG(0xFFB94044);
        uint32_t dropped2 = ETH_READ_REG(0xFFB9404C);

        DPRINT << "tx_cnt: " << tx_cnt2 << " local_rn: " << local_rn2 << ", remote_rn: " << remote_rn2
               << ", dropped: " << dropped2 << ENDL();

    } else {
        uint32_t local_rn0 = ETH_READ_REG(0xFFB94040);
        uint32_t pkt_st_cnt0 = ETH_READ_REG(0xFFB94024);
        uint32_t pkt_end_cnt0 = ETH_READ_REG(0xFFB94028);
        uint32_t low_32bit_received0 = ETH_READ_REG(0xFFB94070);

        DPRINT << "local_rn: " << local_rn0 << ", pkt_st_cnt: " << pkt_st_cnt0 << ", pkt_end_cnt: " << pkt_end_cnt0
               << ", low_32bit_received: " << HEX() << low_32bit_received0 << DEC() << ENDL();

        uint32_t rcvr_rdback = 0;
        while (rcvr_rdback != value) {
            uint32_t without_adjust = NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
            rcvr_rdback = NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX) &
                          ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1);
            if (without_adjust == value) {
                break;
            }
        }

        for (int i = 0; i < 4096; i++) {
            asm("nop");
        }

        uint32_t local_rn1 = ETH_READ_REG(0xFFB94040);
        uint32_t pkt_st_cnt1 = ETH_READ_REG(0xFFB94024);
        uint32_t pkt_end_cnt1 = ETH_READ_REG(0xFFB94028);
        uint32_t low_32bit_received1 = ETH_READ_REG(0xFFB94070);
        uint32_t next_received = ETH_READ_REG(0xFFB94074);
        uint32_t next_received1 = ETH_READ_REG(0xFFB94078);
        uint32_t next_received2 = ETH_READ_REG(0xFFB9407C);
        uint32_t next_received3 = ETH_READ_REG(0xFFB94080);
        uint32_t next_received4 = ETH_READ_REG(0xFFB94084);

        DPRINT << "local_rn: " << local_rn1 << ", pkt_st_cnt: " << pkt_st_cnt1 << ", pkt_end_cnt: " << pkt_end_cnt1
               << ", low_32bit_received: " << HEX() << low_32bit_received1 << " " << next_received << " "
               << next_received1 << " " << next_received2 << " " << next_received3 << " " << next_received4 << DEC()
               << ENDL();

        // *result_addr_ptr = rcvr_rdback;
    }

    if (is_sender) {
        uint32_t tx_cnt3 = ETH_READ_REG(0xFFB90030);
        uint32_t local_rn3 = ETH_READ_REG(0xFFB94040);
        uint32_t remote_rn3 = ETH_READ_REG(0xFFB94044);
        uint32_t dropped3 = ETH_READ_REG(0xFFB9404C);

        DPRINT << "tx_cnt: " << tx_cnt3 << " local_rn: " << local_rn3 << ", remote_rn: " << remote_rn3
               << ", dropped: " << dropped3 << ENDL();
    }
}
