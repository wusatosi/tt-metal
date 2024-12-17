#include "fused_gpt_block.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

namespace {

ttnn::Tensor matmul(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
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

ttnn::Tensor linear(const ttnn::Tensor& tensor, const ttnn::Tensor& weight, const ttnn::Tensor& bias) {
    return ttnn::linear(
        tensor,
        weight,
        bias,
        /* transpose_a */ false,
        /* tranpose_b */ true,
        /* memory_config */ std::nullopt,
        /* dtype */ std::nullopt,
        /* program_config */ std::nullopt,
        /* activation */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::matmul(),
        /* core_grid */ ttnn::CoreGrid{7, 8});
}

}  // namespace

std::pair<ttnn::Tensor, Parameters> layer_norm_forward(
    const ttnn::Tensor& input,
    Parameters& parameters,
    const std::string& layer_name,
    bool return_intermediates = false) {
    auto tensor_shape = input.get_shape();
    auto mean = core::empty(
        core::create_shape({tensor_shape[0], tensor_shape[1], tensor_shape[2], 1}),
        &autograd::ctx().get_device(),
        input.memory_config());
    auto rstd = ttnn::empty_like(mean);
    auto output = ttnn::empty_like(input);

    auto gamma = std::get<autograd::TensorPtr>(parameters[layer_name + ".gamma"]);
    auto beta = std::get<autograd::TensorPtr>(parameters[layer_name + ".beta"]);

    auto out_tensors = ttnn::moreh_layer_norm(
        input,
        1,
        1e-6F,
        /* gamma */ gamma->get_value(),
        /* beta */ beta->get_value(),
        output,
        mean,
        rstd,
        /* memory_config */ std::nullopt,
        /* compute_kernel_config */ std::nullopt);

    assert(out_tensors[0].has_value());
    auto result = out_tensors[0].value();

    Parameters intermediates;
    if (return_intermediates) {
        mean = out_tensors[1].value();
        rstd = out_tensors[2].value();
        intermediates[layer_name + ".mean"] = mean;
        intermediates[layer_name + ".rstd"] = rstd;
    }

    return {result, intermediates};
}

ttnn::Tensor layer_norm_backward(
    const ttnn::Tensor& grad,
    const ttnn::Tensor& input,
    Parameters& parameters,
    const Parameters& intermediates,
    const std::string& layer_name) {
    auto mean = std::get<ttnn::Tensor>(parameters[layer_name + ".mean"]);
    auto rstd = std::get<ttnn::Tensor>(parameters[layer_name + ".rstd"]);
    auto beta = std::get<autograd::TensorPtr>(parameters[layer_name + ".beta"]);
    auto gamma = std::get<autograd::TensorPtr>(parameters[layer_name + ".gamma"]);

    auto input_grad = ttnn::empty_like(input);
    auto gamma_grad = ttnn::empty_like(gamma->get_value());
    auto beta_grad = ttnn::empty_like(beta->get_value());

    auto res = ttnn::moreh_layer_norm_backward(
        grad,
        input,
        mean,
        rstd,
        1,
        gamma->get_value(),
        input_grad,
        gamma_grad,
        beta_grad,
        /* memory_config */ std::nullopt,
        /* compute_kernel_config */ std::nullopt);

    assert(res[0].has_value());
    assert(res[1].has_value());
    assert(res[2].has_value());
    gamma->add_grad(res[1].value());
    beta->add_grad(res[2].value());
    return res[0].value();
}

std::pair<ttnn::Tensor, Parameters> dropout(const ttnn::Tensor& tensor, float probability) {
    if (probability == 0.0F) {
        return {tensor, {}};
    }
    auto mask = core::ones_like(tensor);
    // dropout seed is not properly used in ttnn::dropout
    // auto dropout_seed = autograd::ctx().get_generator()();

    // currently seed is not used in ttnn::dropout
    // we use default seed for now to simplify job of program cache
    // it will require to generate only one program and reuse it later
    auto dropout_seed = 0U;
    auto scaler = 1.0F / (1.0F - probability);
    mask = ttnn::dropout(mask, dropout_seed, probability, scaler);
    auto out = autograd::create_tensor();
    auto masked_out = ttnn::multiply(tensor, mask);

    Parameters intermediates;
    intermediates["mask"] = mask;
    return {masked_out, intermediates};
}

std::pair<ttnn::Tensor, Parameters> multi_head_attention_forward(
    const ttnn::Tensor& input,
    const std::optional<ttnn::Tensor>& mask,
    Parameters& parameters,
    const std::string& layer_name,
    bool return_intermediates = false) {
    auto qkv_weights = std::get<ttnn::Tensor>(parameters[layer_name + ".qkv_weights"]);
    auto qkv_bias = std::get<ttnn::Tensor>(parameters[layer_name + ".qkv_bias"]);
    auto qkv = linear(input, qkv_weights, qkv_bias);

    auto num_heads = std::get<int>(std::get<serialization::ValueType>(parameters[layer_name + ".num_heads"]));

    auto [query, key, value] = ttnn::experimental::nlp_create_qkv_heads(
        qkv,
        std::nullopt,
        num_heads,
        num_heads,
        /* transpose_k */ false,
        /* memory_config */ std::nullopt,
        /* optional_output_tensors */ std::nullopt);

    const float scale = 1.0F / std::sqrtf(static_cast<float>(query.get_shape()[-1]));
    // (B, H, S, E) x (B, H, E, S) -> (B, H, S, S)
    auto qk_t =
        matmul(query, key, /* transpose_a */ false, /* transpose_b */ true, core::ComputeKernelConfig::matmul());
    // (B, H, S, S) * scale
    auto qk_scaled = ttnn::multiply(qk_t, scale);
    if (mask.has_value()) {
        qk_scaled = ttnn::where(mask.value(), qk_scaled, /* other */ -1e9F);
    }
    // (B, H, S, S)
    auto attention_weights = ttnn_fixed::softmax(qk_scaled, /* axis */ 3);

    // (B, H, S, S) x (B, H, S, E) -> (B, H, S, E)
    auto attention_qkv = matmul(
        attention_weights,
        value,
        /* transpose_a */ false,
        /* transpose_b */ false,
        core::ComputeKernelConfig::matmul());

    auto fused_heads = ttnn::experimental::nlp_concat_heads(attention_qkv);

    auto out_linear_weights = std::get<ttnn::Tensor>(parameters[layer_name + ".out_linear_weights"]);
    auto out_linear_bias = std::get<ttnn::Tensor>(parameters[layer_name + ".out_linear_bias"]);
    auto out = linear(fused_heads, out_linear_weights, out_linear_bias);

    auto dropout_prob = std::get<float>(std::get<serialization::ValueType>(parameters[layer_name + ".dropout_prob"]));
    auto [result, dropout_intermediates] = dropout(out, dropout_prob);

    Parameters intermediates;
    intermediates["dropout_mask"] = dropout_intermediates["mask"];
    if (return_intermediates) {
        intermediates["q"] = query;
        intermediates["k"] = key;
        intermediates["v"] = value;
        intermediates["qkv"] = qkv;
        intermediates["qk_t"] = qk_t;
        intermediates["qk_scaled"] = qk_scaled;
        intermediates["attention_weights"] = attention_weights;
        intermediates["attention_qkv"] = attention_qkv;
        intermediates["fused_heads"] = fused_heads;
        intermediates["out"] = out;
    }
    return {result, intermediates};
}

std::pair<ttnn::Tensor, Parameters> mlp_forward(
    const ttnn::Tensor& tensor,
    Parameters& parameters,
    const std::string& layer_name,
    bool return_intermediates = false) {
    auto linear1_weights = std::get<ttnn::Tensor>(parameters[layer_name + ".linear1_weights"]);
    auto linear1_bias = std::get<ttnn::Tensor>(parameters[layer_name + ".linear1_bias"]);

    auto linear2_weights = std::get<ttnn::Tensor>(parameters[layer_name + ".linear2_weights"]);
    auto linear2_bias = std::get<ttnn::Tensor>(parameters[layer_name + ".linear2_bias"]);

    auto fc1 = linear(tensor, linear1_weights, linear1_bias);
    auto gelu = ttnn::gelu(fc1);
    auto fc2 = linear(gelu, linear2_weights, linear2_bias);

    auto dropout_prob = std::get<float>(std::get<serialization::ValueType>(parameters[layer_name + ".dropout_prob"]));
    auto [result, dropout_intermediates] = dropout(fc2, dropout_prob);
    Parameters intermediates;
    intermediates["dropout_mask"] = dropout_intermediates["mask"];
    if (return_intermediates) {
        intermediates["fc1"] = fc1;
        intermediates["gelu"] = gelu;
        intermediates["fc2"] = fc2;
    }
    return {result, intermediates};
}

autograd::TensorPtr fused_gpt_block(
    const autograd::TensorPtr& input, const autograd::TensorPtr& mask, Parameters& parameters) {
    auto [ln1, _] = layer_norm_forward(input->get_value(), parameters, "ln1", false);
    auto [mha, mha_dropout_intermediates] = multi_head_attention_forward(ln1, mask->get_value(), parameters, "mha");
    auto mha_residual = ttnn::add(input->get_value(), mha);

    auto [ln2, _] = layer_norm_forward(mha_residual, parameters, "ln2", false);
    auto [mlp, mlp_dropout_intermediates] = mlp_forward(ln2, parameters, "mlp");
    auto mlp_residual = ttnn::add(mha_residual, mlp);

    auto result = autograd::create_tensor(mlp_residual);

    return result;
}

}  // namespace ttml::ops
