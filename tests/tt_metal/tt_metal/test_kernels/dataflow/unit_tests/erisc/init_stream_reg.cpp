// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    uint32_t init_value = get_arg_val<uint32_t>(0);
    uint32_t result_addr = get_arg_val<uint32_t>(1);

    volatile tt_l1_ptr uint32_t* result_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_addr);

    NOC_STREAM_WRITE_REG(0, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, init_value);

    uint32_t rdbk_val = NOC_STREAM_READ_REG(0, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);

    uint32_t final_result =
        (NOC_STREAM_READ_REG(0, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX) &
         ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1));

    const uint64_t noc_sem_addr = get_noc_addr(27, 25, result_addr, 0);
    // DPRINT << "not stateful api " << HEX() << noc_sem_addr << " value is " << (uint32_t)this->buffer_slot_wrptr <<
    // DEC() << ENDL();
    noc_inline_dw_write(noc_sem_addr, init_value, 0xf, 0);
    noc_async_write_barrier();

    *result_addr_ptr = final_result;
}
