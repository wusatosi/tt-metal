// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_fixture.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include <tt-metalium/test_tiles.hpp>
#include "tests/tt_metal/test_utils/tilization.hpp"
#include "tests/tt_metal/test_utils/print_helpers.hpp"
#include "matmul_test_utils.hpp"

using std::vector;
using namespace tt;
using namespace tt::test_utils;
namespace unit_tests_common::matmul::test_matmul_blackhole_hang {

struct MatmulHangStimuli {
    vector<bfloat16> t;  // Raw tensor values
    vector<uint32_t> a;  // Activations
    vector<uint32_t> w;  // Weights
};

struct MatmulHangConfig {
    uint32_t M, K, N;
    // Whether or not to sync full/half DST between MATH and PACK:
    bool dst_full_sync_en = false;
    string reader_kernel;
    string compute_kernel;
    vector<uint32_t> compute_kernel_args;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
};

void create_test_stimuli(MatmulHangStimuli& stimuli, uint32_t M, uint32_t K, uint32_t N) {
    SHAPE shape = {1, 1, M * 32, K * 32};
    tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(
        shape, tt::deprecated::Initialize::RANDOM, 0, 100, std::chrono::system_clock::now().time_since_epoch().count());
    stimuli.t = tensor.get_values();

    auto activations_tilized = test_utils::tilize(tensor.get_values(), M * 32, K * 32);
    auto activations_tile_layout = convert_to_tile_layout(activations_tilized);
    auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
    auto activations_tile_transposed = transpose_tiles(activations, M, K, 1);
    stimuli.a = activations_tile_transposed;

    // auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32);
    auto identity = std::vector<bfloat16>(K * 32 * N * 32, bfloat16(1.0f));
    auto identity_tilized = test_utils::tilize(identity, K * 32, N * 32);
    auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
    auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
    stimuli.w = weights;
}

// This function creates bit masks to model math fidelity phases. This will mask the result only.
void set_math_fid_masks(uint16_t& math_fid_mask, MathFidelity math_fidelity = MathFidelity::HiFi4) {
    auto arch = get_arch_from_string(get_umd_arch_name());
    switch (math_fidelity) {
        case MathFidelity::HiFi4:
        case MathFidelity::HiFi3: {
            break;
        }
        case MathFidelity::HiFi2:
        case MathFidelity::LoFi: {
            math_fid_mask = (arch == tt::ARCH::GRAYSKULL) ? 0xFFF8 : 0xFFFE;
            break;
        }
        default: {
            TT_THROW("Unsupported MathFidelity={}", math_fidelity);
            break;
        }
    }
}

void matmul_tile(
    DispatchFixture* fixture,
    tt_metal::IDevice* device,
    const MatmulHangConfig& cfg,
    vector<uint32_t> activations,
    vector<uint32_t> weights,
    vector<bfloat16> tensor_vals) {
    tt_metal::Program program = tt_metal::CreateProgram();
    CoreCoord core = {0, 0};

    uint32_t M = cfg.M;
    uint32_t K = cfg.K;
    uint32_t N = cfg.N;
    uint32_t num_tiles_a = M * K;
    uint32_t num_tiles_b = K * N;
    uint32_t num_tiles_out = M * N;
    uint32_t single_tile_size_bfp16b = 2 * 32 * 32;  // Single 32x32 tile size for Float16_b / Uint16
    uint32_t single_tile_size_out0 = single_tile_size_bfp16b;
    const size_t dram_buffer_a_size_bfp16b = num_tiles_a * single_tile_size_bfp16b;
    const size_t dram_buffer_b_size_bfp16b = num_tiles_b * single_tile_size_bfp16b;
    const size_t dram_buffer_size_out0 = num_tiles_out * single_tile_size_out0;

    tt_metal::InterleavedBufferConfig a_dram_config{
        .device = device,
        .size = dram_buffer_a_size_bfp16b,
        .page_size = dram_buffer_a_size_bfp16b,
        .buffer_type = tt_metal::BufferType::DRAM};
    tt_metal::InterleavedBufferConfig b_dram_config{
        .device = device,
        .size = dram_buffer_b_size_bfp16b,
        .page_size = dram_buffer_b_size_bfp16b,
        .buffer_type = tt_metal::BufferType::DRAM};
    tt_metal::InterleavedBufferConfig output_dram_config{
        .device = device,
        .size = dram_buffer_size_out0,
        .page_size = dram_buffer_size_out0,
        .buffer_type = tt_metal::BufferType::DRAM};

    auto src0_dram_buffer = CreateBuffer(a_dram_config);
    auto src1_dram_buffer = CreateBuffer(b_dram_config);
    auto dst_dram_buffer = CreateBuffer(output_dram_config);
    tt::log_info(tt::LogTest, "CreateBuffer");

    uint32_t src0_cb_index = 0;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(dram_buffer_a_size_bfp16b, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size_bfp16b);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(dram_buffer_b_size_bfp16b, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size_bfp16b);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t ouput_cb_index = 16;
    vector<uint32_t> reader_l1_args;
    if (cfg.M > 1 || cfg.N > 1 || cfg.K > 1) {
        uint32_t intermediate_cb_index = 24;
        std::map<uint8_t, tt::DataFormat> partials_and_out_data_format_spec = {
            {ouput_cb_index, tt::DataFormat::Float16_b}, {intermediate_cb_index, tt::DataFormat::Float16_b}};

        CoreRangeSet cores(std::set<CoreRange>{CoreRange(core, core)});
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(dram_buffer_size_out0, partials_and_out_data_format_spec)
                .set_page_size(ouput_cb_index, single_tile_size_out0)
                .set_page_size(intermediate_cb_index, single_tile_size_out0);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        reader_l1_args = {
            src0_dram_buffer->address(),
            0,
            src1_dram_buffer->address(),
            0,
            (std::uint32_t)K,
            (std::uint32_t)M,
            (std::uint32_t)N,
            (std::uint32_t)(M * single_tile_size_bfp16b),
            (std::uint32_t)(N * single_tile_size_bfp16b),
            false};
        tt::log_info(tt::LogTest, "reader_l1_args");
    }
    std::map<string, string> compute_defines;
    compute_defines["PACKER_L1_ACC"] = "1";
    auto mm_reader_kernel = tt_metal::CreateKernel(
        program,
        cfg.reader_kernel,
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto mm_kernel = tt_metal::CreateKernel(
        program,
        cfg.compute_kernel,
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = cfg.math_fidelity,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = cfg.dst_full_sync_en,
            .math_approx_mode = true,
            .compile_args = cfg.compute_kernel_args,
            .defines = compute_defines});

    tt::log_info(tt::LogTest, "CreateKernel");
    fixture->WriteBuffer(device, src0_dram_buffer, activations);
    fixture->WriteBuffer(device, src1_dram_buffer, weights);

    tt_metal::SetRuntimeArgs(program, mm_reader_kernel, core, reader_l1_args);

    tt_metal::SetRuntimeArgs(program, unary_writer_kernel, core, {dst_dram_buffer->address(), 0, num_tiles_out});
    tt::log_info(tt::LogTest, "SetRuntimeArgs");

    fixture->RunProgram(device, program);
    tt::log_info(tt::LogTest, "RunProgram");

    // This is tilized result, will not be modified
    std::vector<uint32_t> result_vec;
    fixture->ReadBuffer(device, dst_dram_buffer, result_vec);
    tt::log_info(tt::LogTest, "ReadBuffer");

    DeallocateBuffer(*src0_dram_buffer);
    DeallocateBuffer(*src1_dram_buffer);
    DeallocateBuffer(*dst_dram_buffer);

    tt::log_info(tt::LogTest, "Math Fidelity = {}, DstSyncFull = {}", cfg.math_fidelity, cfg.dst_full_sync_en);
}
}  // namespace unit_tests_common::matmul::test_matmul_blackhole_hang

using namespace tt::test_utils;
using namespace unit_tests_common::matmul::test_matmul_blackhole_hang;

TEST_F(DispatchFixture, TestMatmulBlackholeHangCombined) {
    // First iteration setup
    uint32_t M1 = 2; /* M = 512 from test_benchmark */
    uint32_t K1 = 4; /* K = 1024 from test_benchmark */
    uint32_t N1 = 4; /* N = 1024 from test_benchmark */
    MatmulHangConfig matmul_config1 = {
        .M = M1,
        .K = K1,
        .N = N1,
        .dst_full_sync_en = false,
        .reader_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_with_bias_blocked.cpp",
        .compute_kernel =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/"
            "bmm_large_block_zm_fused_bias_activation.cpp",
        .compute_kernel_args =
            {
                4,  /* in0_block_w */
                2,  /* in0_num_subblocks */
                8,  /* in0_block_num_tiles */
                4,  /* in0_subblock_num_tiles */
                4,  /* in1_num_subblocks */
                16, /* in1_block_num_tiles */
                4,  /* in1_block_w */
                8,  /* num_blocks_inner_dim */
                1,  /* num_blocks_w_dim */
                1,  /* num_blocks_h_dim */
                1,  /* out_subblock_h */
                1,  /* out_subblock_w */
                1,  /* out_subblock_num_tiles */
                1,  /* batch */
                8,  /* out_block_num_tiles */
                0,  /* untilize_out */
            },
        .math_fidelity = MathFidelity(4)};
    MatmulHangStimuli stimuli1;
    create_test_stimuli(stimuli1, M1, K1, N1);

    // Second iteration setup
    uint32_t M2 = 2; /* M = 512 from test_benchmark */
    uint32_t K2 = 4; /* K = 1024 from test_benchmark */
    uint32_t N2 = 8; /* N = 2048 from test_benchmark */
    MatmulHangConfig matmul_config2 = {
        .M = M2,
        .K = K2,
        .N = N2,
        .dst_full_sync_en = false,
        .reader_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_with_bias_blocked.cpp",
        .compute_kernel =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/"
            "bmm_large_block_zm_fused_bias_activation.cpp",
        .compute_kernel_args =
            {
                4,  /* in0_block_w */
                2,  /* in0_num_subblocks */
                8,  /* in0_block_num_tiles */
                4,  /* in0_subblock_num_tiles */
                8,  /* in1_num_subblocks */
                32, /* in1_block_num_tiles */
                8,  /* in1_block_w */
                8,  /* num_blocks_inner_dim */
                1,  /* num_blocks_w_dim */
                1,  /* num_blocks_h_dim */
                1,  /* out_subblock_h */
                1,  /* out_subblock_w */
                1,  /* out_subblock_num_tiles */
                1,  /* batch */
                16, /* out_block_num_tiles */
                0,  /* untilize_out */
            },
        .math_fidelity = MathFidelity(4)};
    MatmulHangStimuli stimuli2;
    create_test_stimuli(stimuli2, M2, K2, N2);

    // First iteration execution
    matmul_tile(this, devices_.at(0), matmul_config1, stimuli1.a, stimuli1.w, stimuli1.t);
    // Second iteration execution
    matmul_tile(this, devices_.at(0), matmul_config2, stimuli2.a, stimuli2.w, stimuli2.t);
}
