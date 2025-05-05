// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "tt_metal/hw/inc/circular_buffer.h"

#define dump_unpack(a)                                              \
    do {                                                            \
        DPRINT_UNPACK(DPRINT << "UP: " << #a " = " << a << ENDL()); \
    } while (false)
#define dump_pack(a)                                             \
    do {                                                         \
        DPRINT_PACK(DPRINT << "P: " << #a " = " << a << ENDL()); \
    } while (false)
#define dump_math(a)                                             \
    do {                                                         \
        DPRINT_MATH(DPRINT << "M: " << #a " = " << a << ENDL()); \
    } while (false)
namespace NAMESPACE {

void MAIN {
    uint32_t start_block = get_arg_val<uint32_t>(0);
    uint32_t end_block = get_arg_val<uint32_t>(1);

    constexpr auto cb_in = 0;
    constexpr auto cb_out = 1;

    untilize_init(cb_in, cb_out);
    dump_unpack(start_block);
    dump_unpack(end_block);
    for (uint32_t block = start_block; block < end_block; block++) {
        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);
        untilize_block(cb_in, 1, cb_out);
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
    }
    untilize_uninit(cb_in);
}
}  // namespace NAMESPACE
