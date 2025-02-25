// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>
#include "eth_l1_address_map.h"
#include "ethernet/dataflow_api.h"
#include "debug/assert.h"

struct eth_buffer_slot_sync_t {
    volatile uint32_t bytes_sent;
    volatile uint32_t receiver_ack;
    volatile uint32_t src_id;

    uint32_t reserved_2;
};

FORCE_INLINE void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
    }
}

static constexpr uint32_t NUM_CHANNELS = get_compile_time_arg_val(0);

template <bool MEASURE>
FORCE_INLINE void run_loop_iteration(
    const std::array<uint32_t, NUM_CHANNELS>& channel_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_CHANNELS>& channel_sync_addrs,
    uint32_t full_payload_size) {
    if constexpr (MEASURE) {
        // DeviceZoneScopedN("RECEIVER-LOOP-ITER");
        {
            // DeviceZoneScopedN("PING-REPLIES");
            for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
                while (channel_sync_addrs[i]->bytes_sent == 0) {
                    invalidate_l1_cache();
                    // internal_::risc_context_switch();
                }

                channel_sync_addrs[i]->bytes_sent = 0;
                channel_sync_addrs[i]->receiver_ack = 0;

                // wait for txq to be ready, otherwise we'll
                // hit a context switch in the send command
                // while (eth_txq_is_busy()) {
                //     // internal_::risc_context_switch();
                // }

                eth_send_bytes_over_channel_payload_only_unsafe_one_packet(
                    channel_addrs[i], channel_addrs[i], full_payload_size);

                // eth_send_bytes_over_channel_payload_only_unsafe_one_packet(
                //     reinterpret_cast<uint32_t>(channel_sync_addrs[i]),
                //     reinterpret_cast<uint32_t>(channel_sync_addrs[i]), sizeof(eth_channel_sync_t));
            }
        }
    } else {
        {
            for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
                while (channel_sync_addrs[i]->bytes_sent == 0) {
                    invalidate_l1_cache();
                    // internal_::risc_context_switch();
                }

                channel_sync_addrs[i]->bytes_sent = 0;
                channel_sync_addrs[i]->receiver_ack = 0;

                eth_send_bytes_over_channel_payload_only_unsafe_one_packet(
                    channel_addrs[i], channel_addrs[i], full_payload_size);

                // eth_send_bytes_over_channel_payload_only_unsafe_one_packet(
                //     reinterpret_cast<uint32_t>(channel_sync_addrs[i]),
                //     reinterpret_cast<uint32_t>(channel_sync_addrs[i]), sizeof(eth_channel_sync_t));
            }
        }
    }
}

static constexpr uint32_t MAX_CHANNELS = 8;
void kernel_main() {
    uint32_t arg_idx = 0;
    const uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_messages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t message_size = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t full_payload_size = message_size + sizeof(eth_buffer_slot_sync_t);
    const uint32_t full_payload_size_eth_words = full_payload_size >> 4;

    std::array<uint32_t, NUM_CHANNELS> channel_addrs;
    std::array<volatile eth_buffer_slot_sync_t*, NUM_CHANNELS> channel_sync_addrs;
    {
        uint32_t channel_addr = handshake_addr + sizeof(eth_buffer_slot_sync_t);
        for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
            channel_addrs[i] = channel_addr;
            channel_addr += message_size;
            channel_sync_addrs[i] = reinterpret_cast<volatile eth_buffer_slot_sync_t*>(channel_addr);
            channel_sync_addrs[i]->bytes_sent = 0;
            channel_sync_addrs[i]->receiver_ack = 0;
            channel_addr += sizeof(eth_buffer_slot_sync_t);
        }
    }

    // Avoids hang in issue https://github.com/tenstorrent/tt-metal/issues/9963
    for (uint32_t i = 0; i < 2000000000; i++) {
        asm volatile("nop");
    }

    eth_setup_handshake(handshake_addr, false);

    run_loop_iteration<false>(channel_addrs, channel_sync_addrs, full_payload_size);
    {
        DeviceZoneScopedN("MAIN-TEST-BODY");
        uint32_t i = 0;
        for (uint32_t i = 0; i < num_messages; i++) {
            run_loop_iteration<true>(channel_addrs, channel_sync_addrs, full_payload_size);
        }
    }

    // for some reason unknown, not delaying before reset noc counters caused hang. Need investigate.
    for (int i = 0; i < 1000; ++i) {
        asm volatile("nop");
    }
    ncrisc_noc_counters_init();
}
