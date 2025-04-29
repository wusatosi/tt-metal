// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

constexpr uint32_t cb_in = tt::CBIndex::c_0;
constexpr uint32_t cb_out = tt::CBIndex::c_1;
constexpr uint32_t cb_one = tt::CBIndex::c_2;
constexpr uint32_t cb_intermed = tt::CBIndex::c_3;

constexpr uint32_t ONE_TILE{1};
constexpr uint32_t FIRST_TILE{0};

constexpr uint32_t TILE_DEST{0};
constexpr uint32_t TILE_ACC{1};

// TODO(jbbieniekTT): the following functions are to be explained.

FORCE_INLINE uint32_t
get_tile_id(uint32_t i0, uint32_t i1, uint32_t j, uint32_t tiles_per_row, uint32_t PLo, uint32_t PHi, uint32_t HtWt) {
    uint32_t base_tileid = i0 * (tiles_per_row * PHi * HtWt) + i1;
    uint32_t tileid = base_tileid + j * PHi * HtWt;
    return tileid;
}

FORCE_INLINE uint32_t get_first_work_unit(uint32_t core_id, uint32_t all_cores_num, uint32_t all_work_units) {
    return (core_id * all_work_units) / all_cores_num;
}

FORCE_INLINE uint32_t get_last_work_unit(uint32_t core_id, uint32_t all_cores_num, uint32_t all_work_units) {
    return (core_id < (all_cores_num - 1)) ? get_first_work_unit(core_id + 1, all_cores_num, all_work_units)
                                           : all_work_units;
}
