#include <cstdint>
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "compute_kernel_api/reduce.h"
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
                                               // In layernorm_pre_allgather_merge_sharded.cpp
                                               // Assuming:
                                               // - cb_merge (e.g., tt::CBIndex::c_9) contains the partial sums.
    // - cb_reduce (e.g., tt::CBIndex::c_1) contains a tile with the value 1.0f (for summation).
    // - cb_out (e.g., tt::CBIndex::c_14) is the output CB for the final merged sums.
    // - ex2_stride is the tile stride in cb_merge between E(x^2) from different shards.
    // - ex_stride is the tile stride in cb_merge between E(x) from different shards.
    // - dim_shard_factor is the number of shards to merge.

    ACQ();

    // --- Merge E(x**2) ---
    // Initialize reduction for E(x^2). Output will go to dst_reg 0.
    // The source for reduction is cb_merge. The scaler is from cb_reduce.
    reduce_init_delta<false>(cb_merge, cb_reduce, cb_out);  // cb_out here is for config, not direct packing yet

    // Reduce the first partial sum for E(x^2) (from shard 0)
    // Assuming E(x^2) for shard 0 is at tile 0 in cb_merge
    reduce_tile(cb_merge, cb_reduce, 0 /*tile_idx_in_cb_merge*/, 0 /*scaler_tile_idx*/, 0 /*dst_reg_idx*/);

    // Loop through the rest of the shards and accumulate their E(x^2)
    for (uint32_t dim_idx = 1; dim_idx < dim_shard_factor; dim_idx++) {
        uint32_t current_ex2_tile_idx = dim_idx * ex2_stride;
        reduce_tile(cb_merge, cb_reduce, current_ex2_tile_idx, 0 /*scaler_tile_idx*/, 0 /*dst_reg_idx*/);
    }
    // DPRINT << "MC_EX2_MERGED" << ENDL(); // Your DPRINT

    // Normalize E(x**2) - This part can remain similar if dst_reg[0] is the implicit source
    // Ensure mul_tiles_init and mul_tiles use dst_reg[0] (where sum is) as input.
    // If cb_reduce for mul_tiles needs a different scaler (e.g. 1/total_elements), ensure it's populated.
    mul_tiles_init(0 /*src_dst_reg_idx*/, cb_reduce /*scaler_cb_for_norm*/);
    mul_tiles(
        0 /*src_dst_reg_idx*/,
        cb_reduce /*scaler_cb_for_norm*/,
        0 /*scaler_tile_idx*/,
        0 /*dummy?*/,
        0 /*dst_reg_idx*/);
    pack_tile(0 /*dst_reg_idx*/, cb_out, 0 /*output_tile_idx_for_Ex2*/);  // Pack normalized E(x^2)
    reduce_revert_delta(cb_out);  // Revert delta for the E(x^2) reduction config
    // DPRINT << "MC_EX2_NORMALIZED" << ENDL(); // Your DPRINT

#ifndef RMSNORM
    // --- Merge E(x) ---
    // Initialize reduction for E(x). Output will go to dst_reg 1.
    reduce_init_delta<false>(cb_merge, cb_reduce, cb_out);  // cb_out here for config

    // Reduce the first partial sum for E(x) (from shard 0)
    // Assuming E(x) for shard 0 is at tile 1 (or an offset) in cb_merge
    uint32_t first_ex_tile_idx = 1;  // Or some base_ex_offset
    reduce_tile(cb_merge, cb_reduce, first_ex_tile_idx, 0 /*scaler_tile_idx*/, 1 /*dst_reg_idx*/);

    // Loop through the rest of the shards and accumulate their E(x)
    for (uint32_t dim_idx = 1; dim_idx < dim_shard_factor; dim_idx++) {
        uint32_t current_ex_tile_idx = dim_idx * ex_stride + first_ex_tile_idx;  // Adjust based on actual layout
        reduce_tile(cb_merge, cb_reduce, current_ex_tile_idx, 0 /*scaler_tile_idx*/, 1 /*dst_reg_idx*/);
    }

    // Normalize E(x) - Similar to E(x^2), using dst_reg 1
    mul_tiles_init(1 /*src_dst_reg_idx*/, cb_reduce /*scaler_cb_for_norm*/);
    mul_tiles(
        1 /*src_dst_reg_idx*/,
        cb_reduce /*scaler_cb_for_norm*/,
        0 /*scaler_tile_idx*/,
        0 /*dummy?*/,
        1 /*dst_reg_idx*/);
    pack_tile(1 /*dst_reg_idx*/, cb_out, 1 /*output_tile_idx_for_Ex*/);  // Pack normalized E(x)
    reduce_revert_delta(cb_out);                                         // Revert delta for the E(x) reduction config
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
