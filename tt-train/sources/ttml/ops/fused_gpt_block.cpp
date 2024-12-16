#include "fused_gpt_block.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::ops {

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

struct LayerNormBackwardOutput {
    ttnn::Tensor input_grad;
    ttnn::Tensor gamma_grad;
    ttnn::Tensor beta_grad;
};

LayerNormBackwardOutput layer_norm_backward(
    const ttnn::Tensor& grad,
    const ttnn::Tensor& input,
    Parameters& parameters,
    const Parameters& intermediates,
    const std::string& layer_name) {
    auto mean = std::get<autograd::TensorPtr>(parameters[layer_name + ".mean"]);
}

autograd::TensorPtr fused_gpt_block(const autograd::TensorPtr& input, Parameters& parameters) {
}

}  // namespace ttml::ops
