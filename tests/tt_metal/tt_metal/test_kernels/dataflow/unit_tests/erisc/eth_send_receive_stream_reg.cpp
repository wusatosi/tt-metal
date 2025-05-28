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

    // volatile tt_l1_ptr uint32_t* result_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_addr);

    // init stream reg
    NOC_STREAM_WRITE_REG(stream_id, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, 0);

    eth_setup_handshake(handshake_addr, is_sender);

    if (is_sender) {
        while (internal_::eth_txq_is_busy(DEFAULT_ETH_TXQ)) {
        };
        uint32_t addr = STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
        DPRINT << "stream reg addr " << HEX() << addr << ENDL();
        internal_::eth_write_remote_reg_no_txq_check(DEFAULT_ETH_TXQ, addr, 39 << REMOTE_DEST_BUF_WORDS_FREE_INC);

    } else {
        uint32_t rcvr_rdback = 0;
        while (rcvr_rdback != 39) {
            rcvr_rdback = NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX) &
                          ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1);
            DPRINT << "rcvr_rdback: " << rcvr_rdback << ENDL();
        }

        // *result_addr_ptr = rcvr_rdback;
    }
}
