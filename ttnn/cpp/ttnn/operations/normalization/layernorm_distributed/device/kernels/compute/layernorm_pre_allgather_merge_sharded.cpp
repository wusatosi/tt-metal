#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"
#include "compute_kernel_api/bcast.h"
#include "debug/dprint.h"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

namespace NAMESPACE {
void MAIN {
    // Get compile time args
    constexpr uint32_t dim_shard_factor = get_compile_time_arg_val(0);
    constexpr uint32_t num_outputs_per_seq = get_compile_time_arg_val(1);  // 1 for RMSNorm, 2 for LayerNorm

    // Get runtime args
    uint32_t packed_inv_N = get_arg_val<uint32_t>(0);  // Packed bfloat16 1/N normalization factor

    DPRINT << "MC_START" << ENDL();

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_merge = tt::CBIndex::c_9;   // Merge data from reader
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;  // Reduction scalar (1/N)
    constexpr uint32_t cb_out = tt::CBIndex::c_14;    // Output buffer

    // Initialize compute kernel APIs for addition/multiplication
    binary_op_init_common(cb_merge, cb_reduce, cb_out);

    // Calculate partial result offsets based on norm type
#ifdef RMSNORM
    // For rmsnorm: [E(x^2)_dim0, E(x^2)_dim1, ...]
    constexpr uint32_t ex2_stride = 1;
#else
    // For layernorm: [E(x^2)_dim0, E(x)_dim0, E(x^2)_dim1, E(x)_dim1, ...]
    constexpr uint32_t ex2_stride = 2;
    constexpr uint32_t ex_stride = 2;
#endif

    /*
     * Merge E(x**2) partial results using LLK APIs
     */
    // Wait for all partial results to be ready in merge buffer
    uint32_t total_tiles = dim_shard_factor * num_outputs_per_seq;

    cb_wait_front(cb_merge, total_tiles);

    DPRINT << "MC_WAIT_REDUCE_START" << ENDL();
    cb_wait_front(cb_reduce, 1);  // Wait for normalization factor

    DPRINT << "MC_RESERVE_OUTPUT" << ENDL();
    cb_reserve_back(cb_out, num_outputs_per_seq);

    ACQ();

    // Initialize for addition with accumulation to destination
    add_tiles_init(cb_merge, cb_merge, true);

    // Initialize the accumulator with the first dimension's partial result
    add_tiles(cb_merge, cb_merge, 0, 0, 0);

    // Add remaining dimensions' partials using LLK APIs
    for (uint32_t dim_idx = 1; dim_idx < dim_shard_factor; dim_idx++) {
        uint32_t idx = dim_idx * ex2_stride;
        add_tiles(cb_merge, cb_merge, idx, 0, 0);  // Add to the accumulator
    }

    DPRINT << "MC_EX2_MERGED" << ENDL();

    // Normalize by total element count (multiply by 1/N)
    mul_tiles_init(0, cb_reduce);
    mul_tiles(0, cb_reduce, 0, 0, 0);

    // Pack the normalized E(x**2) result
    pack_tile(0, cb_out, 0);

    DPRINT << "MC_EX2_NORMALIZED" << ENDL();

#ifndef RMSNORM
    /*
     * Merge E(x) partial results for layernorm using LLK APIs
     */
    // Initialize for addition with accumulation to destination
    add_tiles_init(cb_merge, cb_merge, true);

    // Initialize the accumulator with the first dimension's partial E(x)
    add_tiles(cb_merge, cb_merge, 1, 1, 1);

    // Add remaining dimensions' partials using LLK APIs
    for (uint32_t dim_idx = 1; dim_idx < dim_shard_factor; dim_idx++) {
        uint32_t idx = dim_idx * ex_stride + 1;
        add_tiles(cb_merge, cb_merge, idx, 1, 1);  // Add to the accumulator
    }

    // Normalize by total element count (multiply by 1/N)
    mul_tiles_init(1, cb_reduce);
    mul_tiles(1, cb_reduce, 0, 0, 1);

    // Pack the normalized E(x) result
    pack_tile(1, cb_out, 1);

#endif

    REL();

    // Push data to the circular buffer for the writer kernel to handle
    cb_push_back(cb_out, num_outputs_per_seq);

    // Pop the consumed input data
    cb_pop_front(cb_merge, total_tiles);
    cb_pop_front(cb_reduce, 1);

    DPRINT << "MC_COMPLETE" << ENDL();
}
}  // namespace NAMESPACE
