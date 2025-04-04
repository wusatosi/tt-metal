// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/bcast.h"

ALWI void ACQ() { ckernel::acquire_dst(); }
ALWI void REL() { ckernel:: release_dst(); }

namespace NAMESPACE {
void MAIN {
    constexpr int onetile = 1;
    uint32_t has_input_grad = get_arg_val<uint32_t>(0);
    uint32_t has_other_grad = get_arg_val<uint32_t>(1);
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(2);

    init_bcast<ELWMUL, BroadcastType::SCALAR>(tt::CBIndex::c_2, tt::CBIndex::c_0, tt::CBIndex::c_16);
    ckernel::cb_wait_front(tt::CBIndex::c_0, onetile);
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        if (has_input_grad) {
            ckernel::cb_wait_front(tt::CBIndex::c_2, onetile);
            ACQ();
            mul_tiles_bcast<BroadcastType::SCALAR>(tt::CBIndex::c_2, tt::CBIndex::c_0, 0, 0, 0);
            ckernel:: pack_tile(0, tt::CBIndex::c_16);
            ckernel::cb_push_back(tt::CBIndex::c_16, onetile);
            ckernel::cb_pop_front(tt::CBIndex::c_2, onetile);
            REL();
        }

        if (has_other_grad) {
            ckernel::cb_wait_front(tt::CBIndex::c_1, onetile);
            ACQ();
            mul_tiles_bcast<BroadcastType::SCALAR>(tt::CBIndex::c_1, tt::CBIndex::c_0, 0, 0, 0);
            ckernel:: pack_tile(0, tt::CBIndex::c_17);
            ckernel::cb_push_back(tt::CBIndex::c_17, onetile);
            ckernel::cb_pop_front(tt::CBIndex::c_1, onetile);
            REL();
        }
    }
}
}  // namespace NAMESPACE
