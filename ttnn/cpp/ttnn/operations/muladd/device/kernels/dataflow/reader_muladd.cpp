#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
// #include "compute_kernel_api/common.h"

// TODO add start addr for each input, and output for multicore

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t src2_addr = get_arg_val<uint32_t>(2);
    uint32_t src3_addr = get_arg_val<uint32_t>(3);
    uint32_t num_output_tiles = get_arg_val<uint32_t>(4);
    uint32_t output_tile_start_id = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_in2 = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_in3 = tt::CBIndex::c_3;

    DataFormat src0_data_format = get_dataformat(cb_id_in0);
    DataFormat src1_data_format = get_dataformat(cb_id_in1);
    DataFormat src2_data_format = get_dataformat(cb_id_in2);
    DataFormat src3_data_format = get_dataformat(cb_id_in3);

    const uint32_t tile_size = get_tile_size(cb_id_in0);

    const InterleavedAddrGenFast<true> s0 = {
        .bank_base_address = src0_addr, .page_size = tile_size, .data_format = src0_data_format};
    const InterleavedAddrGenFast<true> s1 = {
        .bank_base_address = src1_addr, .page_size = tile_size, .data_format = src1_data_format};
    const InterleavedAddrGenFast<true> s2 = {
        .bank_base_address = src2_addr, .page_size = tile_size, .data_format = src2_data_format};
    const InterleavedAddrGenFast<true> s3 = {
        .bank_base_address = src3_addr, .page_size = tile_size, .data_format = src3_data_format};

    constexpr uint32_t onetile = 1;

    uint32_t current_tile = output_tile_start_id;
    for (uint32_t i = 0; i < num_output_tiles; i++) {
        cb_reserve_back(cb_id_in0, onetile);
        auto l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_tile(current_tile, s0, l1_write_addr_in0);

        cb_reserve_back(cb_id_in1, onetile);
        auto l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        noc_async_read_tile(current_tile, s1, l1_write_addr_in1);

        cb_reserve_back(cb_id_in2, onetile);
        auto l1_write_addr_in2 = get_write_ptr(cb_id_in2);
        noc_async_read_tile(current_tile, s2, l1_write_addr_in2);

        cb_reserve_back(cb_id_in3, onetile);
        auto l1_write_addr_in3 = get_write_ptr(cb_id_in3);
        noc_async_read_tile(current_tile, s3, l1_write_addr_in3);

        noc_async_read_barrier();

        cb_push_back(cb_id_in0, onetile);
        cb_push_back(cb_id_in1, onetile);
        cb_push_back(cb_id_in2, onetile);
        cb_push_back(cb_id_in3, onetile);
        current_tile++;
    }
}
