// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ringbuffer cache implementation

#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <optional>
#include <climits>
#include "assert.hpp"

namespace tt::tt_metal {

/*! @brief Ringbuffer cache manager
 *  This class manages the ringbuffer cache for the prefetcher.
 *  It keeps track of the entries in the ringbuffer and their offsets.
 *  The ringbuffer is divided into blocks of size cache_block_sizeB.
 *  The cache manager keeps track of the entries in the ringbuffer and their offsets.
 */
class RingbufferCacheManager {
    friend class RingbufferCacheTestFixture;             // for unit testing purposes
    friend class RingbufferCacheRandomizedTestsFixture;  // for unit testing purposes

public:
    RingbufferCacheManager(int cache_block_sizeB, int cache_size_blocks, int manager_entry_initial_size) :
        cache_block_sizeB_(cache_block_sizeB),
        cache_size_blocks_(cache_size_blocks),
        cache_manager_initial_entry_size_{manager_entry_initial_size},
        manager_{.rb{*this}} {
        TT_ASSERT(cache_size_blocks > 0, "Ringbuffer cache size must be greater than 0");
        TT_ASSERT(cache_block_sizeB > 0, "Ringbuffer cache block size must be greater than 0");
        TT_ASSERT(
            cache_manager_initial_entry_size_ > 0,
            "Ringbuffer cache manager initial entry size ({}) must be greater than 0 ({}, {})",
            cache_manager_initial_entry_size_,
            cache_block_sizeB,
            cache_size_blocks_);
        TT_ASSERT(
            (cache_manager_initial_entry_size_ & (cache_manager_initial_entry_size_ - 1)) == 0,
            "Ringbuffer cache manager initial entry size ({}) is not a power of 2 ({}, {})",
            cache_manager_initial_entry_size_,
            cache_block_sizeB,
            cache_size_blocks_);
    }
    RingbufferCacheManager() = delete;
    RingbufferCacheManager(const RingbufferCacheManager&) = delete;
    RingbufferCacheManager(RingbufferCacheManager&&) = delete;
    RingbufferCacheManager& operator=(const RingbufferCacheManager&) = delete;
    RingbufferCacheManager& operator=(RingbufferCacheManager&&) = delete;
    ~RingbufferCacheManager() = default;

    /*! @brief Reset the cache state */
    void reset() {
        this->manager_.clear();
        std::vector<RingbufferCacheManagerEntry> temp_entry;
        this->manager_.entry.swap(temp_entry);

        std::vector<int32_t> temp_valid;
        this->valid_.swap(temp_valid);
    }

    struct CacheOffset {
        bool is_cached;   // true if the program is already cached
        uint32_t offset;  // offset of the program in the ringbuffer
    };
    /*! @brief Check if the program is present in the ringbuffer
     *  Add a new entry to the ringbuffer manager if not present
     *  @param pgm_id program id
     *  @param lengthB size of program in bytes
     *  @return struct {bool, offset}, where bool::false indicates that the program was added
     */
    std::optional<CacheOffset> get_cache_offset(uint64_t pgm_id, uint32_t lengthB);

    int get_cache_sizeB(void) const { return this->cache_size_blocks_ * this->cache_block_sizeB_; }

private:
    const int cache_block_sizeB_;
    const int cache_size_blocks_;

    /*! @brief cache manager entry */
    const int cache_manager_initial_entry_size_;
    struct RingbufferCacheManagerEntry {
        uint32_t offset;     // offset in ringbuffer
        uint32_t length;     // length of the program in blocks
        uint32_t valid_idx;  // index into the valid_ array

        RingbufferCacheManagerEntry() = default;
        RingbufferCacheManagerEntry(uint32_t off, uint32_t len, uint32_t val) :
            offset(off), length(len), valid_idx(val) {}
    };

    /*! @brief cache manager  */
    struct {
        RingbufferCacheManager& rb;

        std::vector<RingbufferCacheManagerEntry> entry;  // ringbuffer manager entries
        // the following indexes are for the ringbuffer manager, not the ringbuffer
        uint32_t oldest_idx{0};
        uint32_t next_idx{0};
        // the following is saved for convenience
        uint32_t next_block_offset{0};  // offset of the next block to allocate in ringbuffer

        void update_oldest_idx(void) { this->oldest_idx = (this->oldest_idx + 1) & (entry.size() - 1); }
        void update_next_idx(void) {
            auto local_next_idx = this->next_idx + 1;
            // wraparound next_idx if it is at the end and if cache is full
            if (local_next_idx >= entry.size() and this->oldest_idx == 0) [[unlikely]] {
                if (this->next_block_offset != 0) [[likely]] {
                    // if cache offset of oldest index is 0 and next_block_offset is not 0, then cache is partially
                    // filled
                    this->entry.resize(entry.size() * 2);
                }
            }
            this->next_idx = local_next_idx & (entry.size() - 1);
        }
        void clear() {
            this->oldest_idx = 0;
            this->next_idx = 0;
            this->next_block_offset = 0;
        }
    } manager_;

    /*! @brief indexed by program id. Contains index to entry in cache manager if cache hit */
    std::vector<int32_t> valid_;
    constexpr static int32_t invalid_cache_entry_ = -1;

    void add_manager_entry(uint64_t pgm_id, uint32_t offset, uint32_t length);
    void add_manager_entry_no_evict(uint64_t pgm_id, uint32_t offset, uint32_t length);
    void invalidate_manager_entry(void);
    bool invalidate_oldest_until_wraparound(void);
    bool invalidate_sufficient_blocks(int required_space, int offset = 0);
};

}  // namespace tt::tt_metal
