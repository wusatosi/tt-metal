// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    const auto cb_default = tt::CBIndex::c_24;     // FP32
    const auto cb_input_fp32 = tt::CBIndex::c_25;  // default
    const auto cb_output = tt::CBIndex::c_2;

    constexpr int dst0 = 0;
    const auto onetile = 1;

    UNPACK(DPRINT << "DST_ACCUM_MODE " << static_cast<uint32_t>(DST_ACCUM_MODE) << "\n";)

    binary_op_init_common(cb_input_fp32, cb_default, cb_output);

    tile_regs_acquire();
    cb_wait_front(cb_input_fp32, onetile);

    if (true) {  // error
        reconfig_data_format_srca<DST_ACCUM_MODE>(cb_default);
        reconfig_data_format_srca<DST_ACCUM_MODE>(cb_default, cb_input_fp32);
    } else {
        reconfig_data_format_srca<DST_ACCUM_MODE>(cb_input_fp32);
    }

    copy_tile_to_dst_init_short(cb_input_fp32);
    copy_tile(cb_input_fp32, 0, dst0);
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_output, onetile);
    pack_reconfig_data_format(cb_output);
    pack_tile(dst0, cb_output);
    cb_push_back(cb_output, onetile);
    tile_regs_release();
}
}  // namespace NAMESPACE
