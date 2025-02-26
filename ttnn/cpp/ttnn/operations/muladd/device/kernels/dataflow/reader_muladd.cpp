#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint_pages.h"
#include "debug/dprint.h"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint16_t r = 0; r < 32; ++r) {
        DPRINT << (uint)r << " : "
               << TileSlice(
                      cb_id,
                      tile_id,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)0,
                          .w1 = (uint8_t)32,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}
// #include "compute_kernel_api/common.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t src2_addr = get_arg_val<uint32_t>(2);
    uint32_t src3_addr = get_arg_val<uint32_t>(3);
    uint32_t num_output_tiles = get_arg_val<uint32_t>(4);
    uint32_t output_tile_start_id = get_arg_val<uint32_t>(5);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool src2_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool src3_is_dram = get_compile_time_arg_val(3) == 1;

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_in2 = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_in3 = tt::CBIndex::c_3;

    DataFormat src0_data_format = get_dataformat(cb_id_in0);
    DataFormat src1_data_format = get_dataformat(cb_id_in1);
    DataFormat src2_data_format = get_dataformat(cb_id_in2);
    DataFormat src3_data_format = get_dataformat(cb_id_in3);

    const uint32_t tile_size = get_tile_size(cb_id_in0);
    constexpr uint32_t onetile = 1;
    const auto page_size = tile_size;

    // TODO do this individualy for each input if it's sharded
#ifdef IN0_SHARDED
    cb_push_back(tt::CBIndex::c_0, num_output_tiles);
#else
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = page_size, .data_format = src0_data_format};
#endif
#ifdef IN1_SHARDED
    cb_push_back(tt::CBIndex::c_1, num_output_tiles);
#else
    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr, .page_size = page_size, .data_format = src1_data_format};
#endif
#ifdef IN2_SHARDED
    cb_push_back(tt::CBIndex::c_2, num_output_tiles);
#else
    const InterleavedAddrGenFast<src2_is_dram> s2 = {
        .bank_base_address = src2_addr, .page_size = page_size, .data_format = src2_data_format};
#endif
#ifdef IN3_SHARDED
    cb_push_back(tt::CBIndex::c_3, num_output_tiles);
#else
    const InterleavedAddrGenFast<src3_is_dram> s3 = {
        .bank_base_address = src3_addr, .page_size = page_size, .data_format = src3_data_format};
#endif

#if !defined(IN0_SHARDED) || !defined(IN1_SHARDED) || !defined(IN2_SHARDED) || !defined(IN3_SHARDED)
    uint32_t current_tile = output_tile_start_id;
    for (uint32_t i = 0; i < num_output_tiles; i++) {
#ifndef IN0_SHARDED
        cb_reserve_back(cb_id_in0, onetile);
        auto l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_tile(current_tile, s0, l1_write_addr_in0);
#endif
#ifndef IN1_SHARDED
        cb_reserve_back(cb_id_in1, onetile);
        auto l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        noc_async_read_tile(current_tile, s1, l1_write_addr_in1);
#endif

#ifndef IN2_SHARDED
        cb_reserve_back(cb_id_in2, onetile);
        auto l1_write_addr_in2 = get_write_ptr(cb_id_in2);
        noc_async_read_tile(current_tile, s2, l1_write_addr_in2);
#endif

#ifndef IN3_SHARDED
        cb_reserve_back(cb_id_in3, onetile);
        auto l1_write_addr_in3 = get_write_ptr(cb_id_in3);
        noc_async_read_tile(current_tile, s3, l1_write_addr_in3);
#endif
        noc_async_read_barrier();

#ifndef IN0_SHARDED
        cb_push_back(cb_id_in0, onetile);
#endif
#ifndef IN1_SHARDED
        cb_push_back(cb_id_in1, onetile);
#endif
#ifndef IN2_SHARDED
        cb_push_back(cb_id_in2, onetile);
#endif
#ifndef IN3_SHARDED
        cb_push_back(cb_id_in3, onetile);
#endif
        current_tile++;
    }
#endif
}
