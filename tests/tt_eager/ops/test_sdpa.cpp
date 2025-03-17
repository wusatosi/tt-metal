// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/functions.hpp"

using namespace tt;
using namespace tt_metal;
using namespace constants;

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        // Mt, Nt, Kt = num tiles, B = batch
        uint32_t b = 1;
        uint32_t nh = 1;
        uint32_t nkv = 1;
        uint32_t s = 2048;
        uint32_t d = 128;
        uint32_t q_chunk_size = 256;
        uint32_t k_chunk_size = 512;

        ttnn::Shape shape_q({b, nh, s, d});
        ttnn::Shape shape_k({b, nkv, s, d});
        ttnn::Shape shape_v({b, nkv, s, d});

        // Allocates a DRAM buffer on device populated with values specified by initialize
        Tensor q = ttnn::random::random(shape_q).to_layout(Layout::TILE).to_device(device);
        Tensor k = ttnn::random::random(shape_k).to_layout(Layout::TILE).to_device(device);
        Tensor v = ttnn::random::random(shape_v).to_layout(Layout::TILE).to_device(device);

        auto kernel_config = ttnn::WormholeComputeKernelConfig{MathFidelity::HiFi2, true, false, false, false};

        auto program_config = ttnn::operations::transformer::SDPAProgramConfig(
            {1, 1}, std::nullopt, q_chunk_size, k_chunk_size, false, 16);

        Tensor result = ttnn::transformer::scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            /*attn_mask=*/std::nullopt,
                            /*is_causal=*/false,
                            /*scale=*/std::nullopt,
                            /*memory_config=*/std::nullopt,
                            /*program_config=*/program_config,
                            /*compute_kernel_config=*/kernel_config)
                            .cpu();

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");

    return 0;
}
