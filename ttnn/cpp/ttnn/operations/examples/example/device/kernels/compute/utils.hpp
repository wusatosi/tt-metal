#include "compute_kernel_api.h"
#include "tensix_types.h"
#include "tensix.h"
#include "tdma_xmov.h"
#include "debug/dprint.h"

#define ALIGN_MASK 0xF
#define SRAM false

ALWI uint8_t is_unaligned(const uint32_t& src_addr, const uint32_t& dst_addr) {
    return (uint8_t)((src_addr ^ dst_addr) & ALIGN_MASK);
}

ALWI void write_through_mover(uint32_t src_addr, uint32_t dst_addr, uint32_t data_size_bytes) {
    constexpr auto xmov_direction = xmov_direction_t::XMOV_L1_TO_L1;
    // fix mover number as 0 for test
    // maybe exploit two mover later with a proper state machine?
    constexpr auto mover_number = tdma_mover_id_t::TDMA_MOVER0;
    tdma_xmov(mover_number, src_addr, dst_addr, data_size_bytes, xmov_direction);
}

ALWI void wait_write_on_mover() { wait_tdma_movers_done(RISCV_TDMA_STATUS_FLAG_MOVER0_BUSY_MASK); }

ALWI void write_from_packer_to_buffer() {}

ALWI void print_cb_pack(uint32_t cb_id) {
    for (int32_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT_PACK({ DPRINT << (uint)r << " --READ--cin1-- " << TileSlice(cb_id, 0, sr, true, false) << ENDL(); });
    }
}
