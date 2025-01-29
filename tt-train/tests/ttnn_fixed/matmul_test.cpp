// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "init/cpu_initializers.hpp"
#include "init/tensor_initializers.hpp"
#include "ops/linear_op.hpp"
#include "xtensor/xmanipulation.hpp"

class LinearOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

tt::tt_metal::Tensor matmul(
    const tt::tt_metal::Tensor& a,
    const tt::tt_metal::Tensor& b,
    bool transpose_a,
    bool transpose_b,
    const ttnn::WormholeComputeKernelConfig& config) {
    return ttnn::matmul(
        a,
        b,
        transpose_a,
        transpose_b,
        /* memory_config */ std::nullopt,
        /* dtype */ std::nullopt,
        /* program_config */ std::nullopt,
        /* activation */ std::nullopt,
        /* compute_kernel_config */
        config,
        /* core_grid */ ttnn::CoreGrid{7, 8},
        /* output_tile */ std::nullopt);
}

tt::tt_metal::Tensor moreh_matmul(
    const tt::tt_metal::Tensor& a,
    const tt::tt_metal::Tensor& b,
    bool transpose_a,
    bool transpose_b,
    const ttnn::WormholeComputeKernelConfig& config) {
    return ttnn::moreh_matmul(
        a,
        b,
        transpose_a,
        transpose_b,
        /* memory_config */ std::nullopt,
        std::nullopt,
        std::nullopt,
        config);
}

void test_linear(uint32_t batch, uint32_t vocab_size, uint32_t seq_length, uint32_t embedding_dim, bool moreh = false) {
    ttml::autograd::ctx().set_seed(322);

    auto* device = &ttml::autograd::ctx().get_device();
    auto tensor = ttml::autograd::create_tensor();
    ttml::init::normal_init(
        tensor, ttml::core::create_shape({batch, 1, seq_length, embedding_dim}), ttml::init::NormalParams{0.F, 0.02F});

    auto weight = ttml::autograd::create_tensor();
    ttml::init::normal_init(
        weight, ttml::core::create_shape({1, 1, vocab_size, embedding_dim}), ttml::init::NormalParams{0.F, 0.02F});

    auto matmul_l = [&moreh](const tt::tt_metal::Tensor& a, const tt::tt_metal::Tensor& b) {
        if (moreh) {
            // auto b_t = ttnn::transpose(b, -1, -2, b.memory_config());
            return moreh_matmul(a, b, false, true, ttml::core::ComputeKernelConfig::matmul());
        } else {
            return matmul(a, b, false, true, ttml::core::ComputeKernelConfig::matmul());
        }
    };
    auto res = matmul_l(tensor->get_value(), weight->get_value());

    auto x_tensor = ttml::core::to_xtensor(tensor->get_value());
    auto x_weight = ttml::core::to_xtensor(weight->get_value());
    auto x_res = ttml::core::to_xtensor(res);
    x_weight = xt::transpose(x_weight, {0, 1, 3, 2});
    x_weight.reshape({embedding_dim, vocab_size});
    x_tensor.reshape({batch * seq_length, embedding_dim});
    std::cout << "x_tensor shape: " << xt::adapt(x_tensor.shape()) << std::endl;
    std::cout << "x_weight shape: " << xt::adapt(x_weight.shape()) << std::endl;
    xt::xarray<float> golden = xt::linalg::dot(x_tensor, x_weight);
    golden.reshape({batch, 1, seq_length, vocab_size});
    std::cout << "golden shape: " << xt::adapt(golden.shape()) << std::endl;
    for (int i = 0; i < batch; i++) {
        auto x_res_c = xt::view(x_res, i, xt::all(), xt::all(), xt::all());
        auto golden_c = xt::view(golden, i, xt::all(), xt::all(), xt::all());
        auto x_norm_res = xt::sum(xt::pow(x_res_c, 2.0F));
        auto x_norm_golden = xt::sum(xt::pow(golden_c, 2.0F));
        auto diff_norm = xt::sum(xt::pow(x_res_c - golden_c, 2.0F));
        std::cout << "Batch " << i << std::endl;
        std::cout << "norm res: " << x_norm_res << std::endl;
        std::cout << "norm golden: " << x_norm_golden << std::endl;
        std::cout << "diff norm: " << diff_norm << std::endl;
        std::cout << "res min and max " << xt::amin(x_res_c) << " " << xt::amax(x_res_c) << std::endl;
        std::cout << "golden min and max " << xt::amin(golden_c) << " " << xt::amax(golden_c) << std::endl;

        std::cout << x_res_c << std::endl;
        std::cout << golden_c << std::endl;

        EXPECT_TRUE(xt::allclose(x_res_c, golden_c, /*rtol=*/1e-3, /*atol=*/1e-2));
    }
}

TEST_F(LinearOpTest, TTNNLinearOpCrash) {
    uint32_t dim = 32;
    uint32_t batch = 4;  // it works with batch = 1, please try to check from 4 to 64
    uint32_t seq_length = 1024;
    uint32_t embedding_dim = 768;
    bool moreh = false;
    test_linear(batch, dim, seq_length, embedding_dim, moreh);
}
