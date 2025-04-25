// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

constexpr uint32_t cb_in = tt::CB::c_in0;
constexpr uint32_t cb_out = tt::CB::c_out0;
constexpr uint32_t cb_one = tt::CB::c_intermed1;
constexpr uint32_t cb_intermed = tt::CB::c_intermed0;

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

FORCE_INLINE uint32_t get_start_tile_id(uint32_t core_id, uint32_t all_cores_num, uint32_t all_work_units) {
    return (core_id * all_work_units) / all_cores_num;
}

FORCE_INLINE uint32_t get_end_tile_id(uint32_t core_id, uint32_t all_cores_num, uint32_t all_work_units) {
    return (core_id < (all_cores_num - 1)) ? get_start_tile_id(core_id + 1, all_cores_num, all_work_units)
                                           : all_work_units;
}

// FORCE_INLINE std::pair<uint32_t, uint32_t> get_acc_mask(DataFormat df) {
//     std::pair<uint32_t, uint32_t> ret{};
//     switch (df) {
//         //
//     }

//     return ret;
// }
