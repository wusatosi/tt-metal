// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/hw/inc/utils/utils.h"
#include <cstdint>
#include <cstddef>
#include <tuple>
#include "debug/dprint.h"

// Increments val and wraps to 0 if it reaches limit
template <size_t LIMIT = 0, typename T>
FORCE_INLINE auto wrap_increment(T val) -> T {
    constexpr bool is_pow2 = LIMIT != 0 && is_power_of_2(LIMIT);
    if constexpr (LIMIT == 1) {
        return val;
    } else if constexpr (LIMIT == 2) {
        return 1 - val;
    } else if constexpr (is_pow2) {
        return (val + 1) & (static_cast<T>(LIMIT - 1));
    } else {
        return (val == static_cast<T>(LIMIT - 1)) ? static_cast<T>(0) : static_cast<T>(val + 1);
    }
}
template <size_t LIMIT, typename T>
FORCE_INLINE auto wrap_increment_n(T val, uint8_t increment) -> T {
    constexpr bool is_pow2 = LIMIT != 0 && is_power_of_2(LIMIT);
    if constexpr (LIMIT == 1) {
        return val;
    } else if constexpr (LIMIT == 2) {
        return 1 - val;
    } else if constexpr (is_pow2) {
        return (val + increment) & (LIMIT - 1);
    } else {
        T new_unadjusted_val = val + increment;
        bool wraps = new_unadjusted_val >= LIMIT;
        return wraps ? static_cast<T>(new_unadjusted_val - LIMIT) : static_cast<T>(new_unadjusted_val);
    }
}

template <typename T, typename Parameter>
class NamedType {
public:
    FORCE_INLINE explicit NamedType(const T& value) : value_(value) {}
    FORCE_INLINE explicit NamedType(T&& value) : value_(std::move(value)) {}
    FORCE_INLINE NamedType<T, Parameter>& operator=(const NamedType<T, Parameter>& rhs) = default;
    FORCE_INLINE T& get() { return value_; }
    FORCE_INLINE const T& get() const { return value_; }
    FORCE_INLINE operator T() const { return value_; }
    FORCE_INLINE operator T&() { return value_; }

private:
    T value_;
};

using TransactionId = NamedType<uint8_t, struct TransactionIdType>;

template <uint8_t NUM_TRIDS>
struct TransactionIdCounter {
    FORCE_INLINE void increment() { this->next_trid = tt::fabric::wrap_increment<NUM_TRIDS>(this->next_trid); }

    FORCE_INLINE uint8_t get() const { return this->next_trid; }

private:
    uint8_t next_trid = 0;
};

template <size_t NUM_TRIDS>
struct WriteTransactionIdTracker {
    static constexpr uint8_t INVALID_TRID = NUM_TRIDS;
    static constexpr bool N_TRIDS_IS_POW2 = is_power_of_2(NUM_TRIDS);
    static_assert(N_TRIDS_IS_POW2, "NUM_TRIDS must be a power of 2");

    WriteTransactionIdTracker(uint32_t cb_id) :
        cb_interface(get_local_cb_interface(cb_id)),
        trid_counter({}),
        cb_id(cb_id),
        open_trids(0),
        oldest_trid(0),
        next_trid(0) {
        DPRINT << "WriteTransactionIdTracker" << ENDL();
        DPRINT << "\tcb_id: " << (uint32_t)cb_id << "" << ENDL();
        DPRINT << "\toldest_trid: " << (uint32_t)oldest_trid << "" << ENDL();
        DPRINT << "\tnext_trid: " << (uint32_t)next_trid << "" << ENDL();
        DPRINT << "\topen_trids: " << (uint32_t)open_trids << "" << ENDL();
        // Check for invalid usage
    }

    bool oldest_trid_flushed() { return ncrisc_noc_nonposted_write_with_transaction_id_sent(noc_index, oldest_trid); }

    bool cb_next_slot_is_available(size_t num_pages) {
        return cb_pages_available_at_front(cb_id, total_pages_open + num_pages);
    }

    bool backpressured() { return open_trids == NUM_TRIDS; }

    // How does this work with wrapping?
    // Do I need to know the CB ID too?
    std::tuple<size_t, TransactionId> get_next_cb_slot(size_t num_pages) {
        auto chunk_offset_from_rdptr =
            open_trids * static_cast<uint32_t>(total_pages_open) * cb_interface.fifo_page_size;

        auto non_wrapped_addr = cb_interface.fifo_rd_ptr + chunk_offset_from_rdptr;
        bool wraps = non_wrapped_addr >= cb_interface.fifo_limit;
        auto chunk_start_address = non_wrapped_addr - (wraps * cb_interface.fifo_size);
        open_trids++;
        DPRINT << "get_next_cb_slot" << ENDL();
        DPRINT << "\tchunk_offset_from_rdptr: " << (uint32_t)chunk_offset_from_rdptr << "" << ENDL();
        DPRINT << "\tnon_wrapped_addr: " << (uint32_t)non_wrapped_addr << "" << ENDL();
        DPRINT << "\twraps: " << (uint32_t)wraps << "" << ENDL();
        DPRINT << "\tchunk_start_address: " << (uint32_t)chunk_start_address << "" << ENDL();
        DPRINT << "\topen_trids: " << (uint32_t)open_trids << "" << ENDL();
        auto next_available_trid = next_trid;
        pages_per_trid[next_available_trid] = num_pages;
        DPRINT << "\t\tnext_available_trid: " << (uint32_t)next_available_trid << "" << ENDL();
        DPRINT << "\t\tpages_per_trid: " << (uint32_t)pages_per_trid[next_available_trid] << "" << ENDL();
        total_pages_open += num_pages;
        DPRINT << "\t\ttotal_pages_open: " << (uint32_t)total_pages_open << "" << ENDL();
        DPRINT << "\tcb_wait_front..." << ENDL();
        cb_wait_front(cb_id, total_pages_open);
        DPRINT << "\t\tdone" << ENDL();
        next_trid = TransactionId{wrap_increment<NUM_TRIDS>(next_trid.get())};
        DPRINT << "\tnext_trid: " << (uint32_t)next_trid << "" << ENDL();
        DPRINT << "done" << ENDL();
        return {chunk_start_address, next_available_trid};
    }

    FORCE_INLINE bool has_unflushed_trid() { return open_trids > 0; }

    FORCE_INLINE void pop_pages_for_oldest_trid() {
        DPRINT << "pop_pages_for_oldest_trid" << ENDL();
        auto pages_to_pop = pages_per_trid[oldest_trid];
        DPRINT << "\tpages_to_pop: " << (uint32_t)pages_to_pop << "" << ENDL();
        cb_pop_front(cb_id, pages_to_pop);
        total_pages_open -= pages_to_pop;
        oldest_trid = TransactionId{wrap_increment<NUM_TRIDS>(oldest_trid.get())};
        open_trids--;
        DPRINT << "\toldest_trid: " << (uint32_t)oldest_trid << "" << ENDL();
        DPRINT << "\topen_trids: " << (uint32_t)open_trids << "" << ENDL();
        DPRINT << "tdone" << ENDL();
    }

    FORCE_INLINE void write_barrier() {
        DPRINT << "write_barrier" << ENDL();
        DPRINT << "\topen_trids: " << (uint32_t)open_trids << "" << ENDL();
        while (open_trids > 0) {
            this->pop_pages_for_oldest_trid();
        }
        DPRINT << "done" << ENDL();
    }

private:
    std::array<size_t, NUM_TRIDS> pages_per_trid;
    LocalCBInterface& cb_interface;
    uint16_t total_pages_open = 0;
    // uint8_t cb_n_chunks = 0;

    // Advances with oldest_trid
    TransactionIdCounter<NUM_TRIDS> trid_counter;
    uint8_t cb_id = 0;
    uint8_t open_trids = 0;

    // TODO: cleanup - only used for when both params are pow2, else above are used.
    TransactionId oldest_trid = 0;
    TransactionId next_trid = 0;
};
