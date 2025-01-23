// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "dprint.h"

namespace NAMESPACE {

void MAIN {
    constexpr auto cb_input = tt::CBIndex::c_0;   // float32
    constexpr auto cb_output = tt::CBIndex::c_3;  // bfloat16

    binary_op_init_common(cb_input, cb_output);

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_input, onetile);

    DPRINT << FIXED() << SETPRECISION(10);
    UNPACK(DPRINT << "DST_ACCUM_MODE " << static_cast<uint32_t>(DST_ACCUM_MODE) << "\n";)

    UNPACK(DPRINT << "AAAA cb_input "
                  << TSLICE(cb_input, 0, SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1}) << ENDL();)

    tile_regs_acquire();
    cb_wait_front(cb_input, onetile);

    reconfig_data_format_srca<true>(cb_input);
    copy_tile_init(cb_input);
    copy_tile(cb_input, 0, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_output);
    pack_tile(dst0, cb_output);
    tile_regs_release();

    cb_push_back(cb_output, onetile);

    cb_wait_front(cb_output, onetile);
    UNPACK(DPRINT << "AAAA cb_output "
                  << TSLICE(cb_output, 0, SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 3, .ws = 1}) << ENDL();)
}
}  // namespace NAMESPACE
