// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "eth_chan_noc_mapping.h"

void kernel_main() {
    uint32_t eth_addr = get_arg_val<uint32_t>(0);

    eth_chan_to_noc_xy[0][0] = (((25 << NOC_ADDR_NODE_ID_BITS) | 20) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[0][1] = (((25 << NOC_ADDR_NODE_ID_BITS) | 21) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[0][2] = (((25 << NOC_ADDR_NODE_ID_BITS) | 22) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[0][3] = (((25 << NOC_ADDR_NODE_ID_BITS) | 23) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[0][4] = (((25 << NOC_ADDR_NODE_ID_BITS) | 24) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[0][5] = (((25 << NOC_ADDR_NODE_ID_BITS) | 25) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[0][6] = (((25 << NOC_ADDR_NODE_ID_BITS) | 26) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[0][7] = (((25 << NOC_ADDR_NODE_ID_BITS) | 27) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[0][8] = (((25 << NOC_ADDR_NODE_ID_BITS) | 28) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[0][9] = (((25 << NOC_ADDR_NODE_ID_BITS) | 29) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[0][10] = (((25 << NOC_ADDR_NODE_ID_BITS) | 30) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[0][11] = (((25 << NOC_ADDR_NODE_ID_BITS) | 31) << NOC_COORD_REG_OFFSET);

    eth_chan_to_noc_xy[1][0] = (((25 << NOC_ADDR_NODE_ID_BITS) | 20) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[1][1] = (((25 << NOC_ADDR_NODE_ID_BITS) | 21) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[1][2] = (((25 << NOC_ADDR_NODE_ID_BITS) | 22) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[1][3] = (((25 << NOC_ADDR_NODE_ID_BITS) | 23) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[1][4] = (((25 << NOC_ADDR_NODE_ID_BITS) | 24) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[1][5] = (((25 << NOC_ADDR_NODE_ID_BITS) | 25) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[1][6] = (((25 << NOC_ADDR_NODE_ID_BITS) | 26) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[1][7] = (((25 << NOC_ADDR_NODE_ID_BITS) | 27) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[1][8] = (((25 << NOC_ADDR_NODE_ID_BITS) | 28) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[1][9] = (((25 << NOC_ADDR_NODE_ID_BITS) | 29) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[1][10] = (((25 << NOC_ADDR_NODE_ID_BITS) | 30) << NOC_COORD_REG_OFFSET);
    eth_chan_to_noc_xy[1][11] = (((25 << NOC_ADDR_NODE_ID_BITS) | 31) << NOC_COORD_REG_OFFSET);

    for (uint32_t i = 0; i < 12; i++) {
        uint32_t eth_noc_xy = eth_chan_to_noc_xy[0][i];
        uint64_t noc_sem_addr = get_noc_addr_helper(eth_noc_xy, eth_addr);
        DPRINT << "Writing to " << HEX() << noc_sem_addr << DEC() << ENDL();
        noc_inline_dw_write(noc_sem_addr, 39);
    }

    noc_async_write_barrier();
}
