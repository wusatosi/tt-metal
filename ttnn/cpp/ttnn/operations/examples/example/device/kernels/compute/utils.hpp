#include "compute_kernel_api.h"
#include "tensix_types.h"
#include "tensix.h"
#include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"

#define ALIGN_MASK 0xF
#define ALIGNED_16B 0x10
#define CMD_TDMA_XMOV 0x40
#define SRAM false

ALWI uint8_t is_unaligned(const uint32_t& src_addr, const uint32_t& dst_addr) {
    return (uint8_t)((src_addr ^ dst_addr) & ALIGN_MASK);
}

ALWI void write_through_pack_tile(uint dst_id, uint cb_id) {
    // DeviceZoneScopedN("write through pack_tile");
    tile_regs_wait();
    pack_tile(dst_id, cb_id, dst_id);
    tile_regs_release();
}

struct Mover {
    const xmov_direction_t transfer_direction = xmov_direction_t::XMOV_L1_TO_L1;
    const tdma_mover_id_t mover_number = tdma_mover_id_t::TDMA_MOVER0;  // fix temporalily
    const uint op_binary = 0x40;
    const uint write_cmd = op_binary | (mover_number << 8);

    void write_field(uint addr, uint value) {
        volatile uint* buf = reinterpret_cast<volatile uint*>(addr);
        buf[0] = value;
    }

    uint read_field(uint addr) {
        volatile uint* buf = reinterpret_cast<volatile uint*>(addr);
        return buf[0];
    }

    bool sanitize(uint src_addr, uint dst_addr) {
        return !is_unaligned(src_addr, ALIGNED_16B) and !is_unaligned(dst_addr, ALIGNED_16B);
    }

    void configure(uint32_t src_addr, uint32_t dst_addr, uint32_t buffer_size) {
        write_field(RISCV_TDMA_REG_XMOV_DST_ADDR, (dst_addr / 16));
        write_field(RISCV_TDMA_REG_XMOV_SRC_ADDR, (src_addr / 16));
        write_field(RISCV_TDMA_REG_XMOV_SIZE, (buffer_size / 16));
        write_field(RISCV_TDMA_REG_XMOV_DIRECTION, (uint)transfer_direction);
    }

    void run() { write_field(RISCV_TDMA_REG_COMMAND_ADDR, write_cmd); }

    void wait() {
        volatile uint tdma_mover_status;
        tdma_mover_status = read_field(RISCV_TDMA_REG_STATUS);

        do {
            tdma_mover_status = read_field(RISCV_TDMA_REG_STATUS);
        } while (
            (tdma_mover_status & (RISCV_TDMA_STATUS_FLAG_MOVER0_BUSY_MASK | RISCV_TDMA_STATUS_FLAG_FIFO_EMPTY_MASK)) !=
            RISCV_TDMA_STATUS_FLAG_FIFO_EMPTY_MASK);
    }
};

ALWI void print_cb_pack(uint32_t cb_id) {
    for (int32_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT_PACK({ DPRINT << (uint)r << " --READ--cin1-- " << TileSlice(cb_id, 0, sr, true, false) << ENDL(); });
    }
}
