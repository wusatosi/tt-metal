// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dprint.h"

#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {

void MAIN {
    constexpr auto cb_scalar = tt::CBIndex::c_2;
    constexpr auto cb_tmp = tt::CBIndex::c_3;

    binary_op_init_common(tt::CBIndex::c_0,  tt::CBIndex::c_1); // incorrect output 4
    // binary_op_init_common(cb_scalar,  cb_scalar, cb_tmp); // correct output 4.02734

    constexpr int dst0 = 0;

    cb_reserve_back(cb_tmp, 1);
    cb_wait_front(cb_scalar, 2);

    tile_regs_acquire();
    reconfig_data_format(cb_scalar, cb_scalar);
    add_tiles_init(cb_scalar, cb_scalar);
    add_tiles(cb_scalar, cb_scalar, 0, 1, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_tmp);
    pack_tile(dst0, cb_tmp);
    tile_regs_release();

    cb_push_back(cb_tmp, 1);

    cb_wait_front(cb_tmp, 1);
    UNPACK(
        DPRINT << "OUTPUT TEST " << TSLICE(cb_tmp, 0, SliceRange{ .h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1 });
    )

}
}  // namespace NAMESPACE
