#include <cstdint>
#include <cstring>
#include "debug/dprint.h"

#define ALIGN_MASK 0xF
#define SRAM false

namespace {
#define ALWI inline __attribute__((always_inline))

ALWI void write_through_memcpy(uint32_t dst_addr, uint32_t src_addr, uint32_t data_size_bytes) {
    std::memcpy(reinterpret_cast<uint32_t*>(dst_addr), reinterpret_cast<uint32_t*>(src_addr), data_size_bytes);
}

ALWI uint8_t is_unaligned(const uint32_t& src_addr, const uint32_t& dst_addr) {
    return (uint8_t)((src_addr ^ dst_addr) & ALIGN_MASK);
}

ALWI void print_read_tile_nc(uint32_t cb_id) {
    for (int32_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT_DATA0({
            DPRINT << (uint)r << " --READ--cin1-- "
                   << TileSlice(cb_id, 0, sr, TSLICE_INPUT_CB, TSLICE_RD_PTR, true, true) << ENDL();
        });
    }
    DPRINT << ENDL();
}

ALWI void print_write_tile_nc(uint32_t cb_id) {
    for (int32_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT_DATA0({
            DPRINT << (uint)r << " --READ--cin1-- "
                   << TileSlice(cb_id, 0, sr, TSLICE_OUTPUT_CB, TSLICE_WR_PTR, true, true) << ENDL();
        });
    }
    DPRINT << ENDL();
}

ALWI void print_read_tile_br(uint32_t cb_id) {
    for (int32_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT_DATA1({
            DPRINT << (uint)r << " --READ--cin1-- "
                   << TileSlice(cb_id, 0, sr, TSLICE_INPUT_CB, TSLICE_RD_PTR, true, true) << ENDL();
        });
    }
    DPRINT << ENDL();
}

ALWI void print_write_tile_br(uint32_t cb_id) {
    for (int32_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT_DATA1({
            DPRINT << (uint)r << " --READ--cin1-- "
                   << TileSlice(cb_id, 0, sr, TSLICE_OUTPUT_CB, TSLICE_WR_PTR, true, true) << ENDL();
        });
    }
    DPRINT << ENDL();
}
}  // namespace
