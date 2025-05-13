#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"
#include "compute_kernel_api/bcast.h"
#include "debug/dprint.h"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

namespace NAMESPACE {
void MAIN {
    DPRINT << "MC_KRNL_START" << ENDL();
    // Get compile time args
    constexpr uint32_t dim_shard_factor = get_compile_time_arg_val(0);
    constexpr uint32_t num_outputs_per_seq = get_compile_time_arg_val(1);

    // Get runtime args
    // uint32_t packed_inv_N = get_arg_val<uint32_t>(0); // Appears unused in this kernel?

    // DPRINT << "MC_ARGS dim_shard=" << dim_shard_factor << " outputs=" << num_outputs_per_seq << ENDL();

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_merge = tt::CBIndex::c_9;   // Merge data from reader
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;  // Reduction scalar (1/N) - Used as input
    constexpr uint32_t cb_out = tt::CBIndex::c_14;    // Output buffer

    binary_op_init_common(cb_merge, cb_reduce, cb_out);

#ifdef RMSNORM
    constexpr uint32_t results_per_dim = 1;
    constexpr uint32_t ex2_stride = 1;
#else
    constexpr uint32_t results_per_dim = 2;
    constexpr uint32_t ex2_stride = 2;
    constexpr uint32_t ex_stride = 2;
#endif

    // uint32_t total_tiles = dim_shard_factor * num_outputs_per_seq; // INCORRECT calculation
    // num_outputs_per_seq is ALREADY the total number of tiles from all dim shards for this sequence
    const uint32_t total_tiles_to_wait = num_outputs_per_seq;  // Use the compile arg directly

    // Wait for data from reader and scalar
    DPRINT << "MC_WAIT_READER target=" << total_tiles_to_wait << ENDL();
    cb_wait_front(cb_merge, total_tiles_to_wait);
    DPRINT << "MC_WAIT_SCALAR" << ENDL();
    cb_wait_front(cb_reduce, 1);  // Wait for normalization factor
    DPRINT << "MC_INPUTS_READY" << ENDL();

    cb_reserve_back(cb_out, results_per_dim);  // Reserve space for final output (1 or 2 tiles)
    ACQ();

    // Merge E(x**2)
    add_tiles_init(cb_merge, cb_merge, true);
    add_tiles(cb_merge, cb_merge, 0, 0, 0);
    for (uint32_t dim_idx = 1; dim_idx < dim_shard_factor; dim_idx++) {
        uint32_t idx = dim_idx * ex2_stride;
        add_tiles(cb_merge, cb_merge, idx, 0, 0);
    }
    // DPRINT << "MC_EX2_MERGED" << ENDL();

    // Normalize E(x**2)
    mul_tiles_init(0, cb_reduce);
    mul_tiles(0, cb_reduce, 0, 0, 0);
    pack_tile(0, cb_out, 0);  // Pack normalized E(x^2) to output buffer
    // DPRINT << "MC_EX2_NORMALIZED" << ENDL();

#ifndef RMSNORM
    // Merge E(x)
    add_tiles_init(cb_merge, cb_merge, true);
    add_tiles(cb_merge, cb_merge, 1, 1, 1);
    for (uint32_t dim_idx = 1; dim_idx < dim_shard_factor; dim_idx++) {
        uint32_t idx = dim_idx * ex_stride + 1;
        add_tiles(cb_merge, cb_merge, idx, 1, 1);
    }

    // Normalize E(x)
    mul_tiles_init(1, cb_reduce);
    mul_tiles(1, cb_reduce, 0, 0, 1);
    pack_tile(1, cb_out, 1);  // Pack normalized E(x) to output buffer
#endif

    REL();

    DPRINT << "MC_PUSH_OUTPUT tiles=" << results_per_dim << ENDL();
    cb_push_back(cb_out, results_per_dim);  // Push final output (1 or 2 tiles)

    // Pop consumed inputs
    cb_pop_front(cb_merge, total_tiles_to_wait);  // Pop the correct total waited tiles
    cb_pop_front(cb_reduce, 1);

    DPRINT << "MC_KRNL_END" << ENDL();
}
}  // namespace NAMESPACE
