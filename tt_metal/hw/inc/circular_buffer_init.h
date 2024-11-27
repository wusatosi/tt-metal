// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "circular_buffer.h"
#include "risc_attribs.h"

#include "remote_circular_buffer_api.h"

// NCRISC and BRISC setup read and write
// TRISC sets up read or write
inline void setup_local_cb_read_write_interfaces(
    uint32_t tt_l1_ptr* cb_l1_base,
    uint32_t start_cb_index,
    uint32_t max_cb_index,
    bool read,
    bool write,
    bool init_wr_tile_ptr) {
    volatile tt_l1_ptr uint32_t* circular_buffer_config_addr =
        cb_l1_base + start_cb_index * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG;

    for (uint32_t cb_id = start_cb_index; cb_id < max_cb_index; cb_id++) {
        // NOTE: fifo_addr, fifo_size and fifo_limit in 16B words!
        uint32_t fifo_addr = circular_buffer_config_addr[0];
        uint32_t fifo_size = circular_buffer_config_addr[1];
        uint32_t fifo_num_pages = circular_buffer_config_addr[2];
        uint32_t fifo_page_size = circular_buffer_config_addr[3];
        uint32_t fifo_limit = fifo_addr + fifo_size;

        LocalCBInterface& local_interface = get_local_cb_interface(cb_id);
        local_interface.fifo_limit = fifo_limit;  // to check if we need to wrap
        if (write) {
            local_interface.fifo_wr_ptr = fifo_addr;
        }
        if (read) {
            local_interface.fifo_rd_ptr = fifo_addr;
        }
        local_interface.fifo_size = fifo_size;
        local_interface.tiles_acked_received_init = 0;
        if (write) {
            local_interface.fifo_num_pages = fifo_num_pages;
        }
        local_interface.fifo_page_size = fifo_page_size;

        if (init_wr_tile_ptr) {
            local_interface.fifo_wr_tile_ptr = 0;
        }

        circular_buffer_config_addr += UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG;
    }
}

namespace experimental {

template <bool update_remote_over_noc = false>
inline void setup_remote_cb_interfaces(uint32_t tt_l1_ptr* cb_l1_base, uint32_t start_cb_index) {
#ifdef COMPILE_FOR_TRISC
    uint8_t noc = 0;
#else
    uint8_t noc = noc_index;
#endif
    volatile tt_l1_ptr uint32_t* circular_buffer_config_addr = cb_l1_base;

    for (uint32_t cb_id = NUM_CIRCULAR_BUFFERS - 1, end_id = start_cb_index - 1; cb_id != end_id; cb_id--) {
        uint32_t config_addr = circular_buffer_config_addr[0];
        uint32_t page_size = circular_buffer_config_addr[1];
        volatile tt_l1_ptr uint32_t* l1_remote_cb_config_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_addr);
        const bool is_sender = l1_remote_cb_config_addr[0];
        uint32_t num_receivers = l1_remote_cb_config_addr[1];
        uint32_t fifo_start_addr = l1_remote_cb_config_addr[2];
        uint32_t fifo_size = l1_remote_cb_config_addr[3];
        uint32_t fifo_ptr = l1_remote_cb_config_addr[4];
        uint32_t remote_noc_xy_addr = l1_remote_cb_config_addr[5];
        uint32_t aligned_pages_sent_addr = l1_remote_cb_config_addr[6];
        if (is_sender) {
            uint32_t aligned_pages_acked_addr = aligned_pages_sent_addr + num_receivers * L1_ALIGNMENT;
            detail::resize_remote_sender_cb_interface<update_remote_over_noc>(
                cb_id,
                fifo_size,
                page_size,
                fifo_start_addr,
                fifo_ptr,
                aligned_pages_sent_addr,
                remote_noc_xy_addr,
                num_receivers,
                noc);
            RemoteSenderCBInterface& sender_cb_interface = get_remote_sender_cb_interface(cb_id);
            sender_cb_interface.config_ptr = config_addr;
            sender_cb_interface.fifo_start_addr = fifo_start_addr;
            sender_cb_interface.receiver_noc_xy_ptr = remote_noc_xy_addr;
            sender_cb_interface.aligned_pages_sent_ptr = aligned_pages_sent_addr;
            sender_cb_interface.num_receivers = num_receivers;
        } else {
            uint32_t aligned_pages_acked_addr = aligned_pages_sent_addr + num_receivers * L1_ALIGNMENT;
            uint32_t sender_noc_x = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_noc_xy_addr)[0];
            uint32_t sender_noc_y = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_noc_xy_addr)[1];
            detail::resize_remote_receiver_cb_interface<update_remote_over_noc>(
                cb_id,
                fifo_size,
                page_size,
                fifo_start_addr,
                fifo_ptr,
                aligned_pages_acked_addr,
                sender_noc_x,
                sender_noc_y,
                noc);
            RemoteReceiverCBInterface& receiver_cb_interface = get_remote_receiver_cb_interface(cb_id);
            receiver_cb_interface.config_ptr = config_addr;
            receiver_cb_interface.fifo_start_addr = fifo_start_addr;
            receiver_cb_interface.sender_noc_x = sender_noc_x;
            receiver_cb_interface.sender_noc_y = sender_noc_y;
            receiver_cb_interface.aligned_pages_sent_ptr = aligned_pages_sent_addr;
            receiver_cb_interface.aligned_pages_acked_ptr = aligned_pages_acked_addr;
        }
        circular_buffer_config_addr += UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG;
    }
}

template <uint32_t num_local_cbs>
FORCE_INLINE void align_local_cbs_to_remote_cb(
    uint32_t remote_cb_index, const uint32_t (&local_cb_indices)[num_local_cbs]) {
    // We assert that the offset of sender and receiver common attributes are the same
    // so we can use either interface here
    const RemoteReceiverCBInterface& remote_cb = get_remote_receiver_cb_interface(remote_cb_index);
    uint32_t fifo_limit = remote_cb.fifo_limit_page_aligned >> 4;
    uint32_t fifo_size = fifo_limit - (remote_cb.fifo_start_addr >> 4);
    uint32_t fifo_ptr = remote_cb.fifo_rd_ptr >> 4;
    for (uint32_t i = 0; i < num_local_cbs; i++) {
        LocalCBInterface& local_cb = get_local_cb_interface(local_cb_indices[i]);
        uint32_t fifo_num_pages = fifo_size / local_cb.fifo_page_size;
        local_cb.fifo_limit = fifo_limit;
        local_cb.fifo_size = fifo_size;
        local_cb.fifo_num_pages = fifo_num_pages;
        local_cb.fifo_wr_ptr = fifo_ptr;
        local_cb.fifo_rd_ptr = fifo_ptr;
    }
}

template <uint32_t num_remote_cbs>
FORCE_INLINE void update_remote_cb_configs_in_l1(const uint32_t (&remote_cb_indices)[num_remote_cbs]) {
    for (auto cb_id : remote_cb_indices) {
        // We assert that the offset of sender fifo_wr_ptr and receiver fifo_rd_ptr are the same
        // so just update the fifo_ptr using either interface
        RemoteReceiverCBInterface& remote_cb_interface = get_remote_receiver_cb_interface(cb_id);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            remote_cb_interface.config_ptr + offsetof(RemoteReceiverCBInterface, fifo_rd_ptr)) =
            remote_cb_interface.fifo_rd_ptr;
    }
}

}  // namespace experimental
