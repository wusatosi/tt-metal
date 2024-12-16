// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "losses.hpp"

#include <core/ttnn_all_includes.hpp>
#include <ttnn/types.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr mse_loss(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target, ReduceType reduce) {
    auto difference = ops::sub(target, prediction);  // TODO: @rfurko-tt use "ttnn::squared_difference"
    auto squared_difference =
        ops::mul(difference, difference);  // TODO: need to add backward "ttnn::squared_difference_bw" might be faster
    if (reduce == ReduceType::MEAN) {
        return ops::mean(squared_difference);
    } else {
        throw std::logic_error("Unsupported MSE reduction type");
    }
}

autograd::TensorPtr cross_entropy_loss_without_reduce_(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target) {
    const float eps = 1e-6F;
    auto prediction_tensor = ttnn_fixed::softmax(prediction->get_value(), 3);
    auto prediction_tensor_clipped = ttnn::clip(prediction_tensor, eps, 1.0F);
    auto loss = ttnn::multiply(target->get_value(), ttnn::log(prediction_tensor_clipped));
    loss = ttnn::neg(loss);
    loss = ttnn_fixed::sum_over_dim(loss, 3);
    auto out = autograd::create_tensor(loss);

    autograd::GradFunction grad = [target, prediction_tensor, prediction, out]() {
        auto grad = ttnn::subtract(prediction_tensor, target->get_value());
        grad = ttnn::multiply(grad, out->get_grad());
        prediction->add_grad(grad);
    };

    auto links = autograd::get_links(prediction);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

xt::xarray<float> log_softmax(const xt::xarray<float>& logits) {
    auto max_vals = xt::amax(logits, {1});
    std::cout << "max_vals shape: " << xt::adapt(max_vals.shape()) << std::endl;
    auto max_vals_2d = xt::view(max_vals, xt::all(), xt::newaxis());
    auto shifted = logits - max_vals_2d;
    auto logsumexp = xt::log(xt::sum(xt::exp(shifted), {1}));
    auto logsumexp_2d = xt::view(logsumexp, xt::all(), xt::newaxis());
    auto result = shifted - logsumexp_2d;
    return result;
}

// Compute softmax from logits
xt::xarray<float> softmax(const xt::xarray<float>& logits) {
    auto lsm = log_softmax(logits);
    return xt::exp(lsm);
}

xt::xarray<float> one_hot(size_t n_classes, const xt::xarray<int>& indices) {
    // Indices should be 1D: shape = [N]
    // The output shape will be [N, n_classes].
    auto ishape = indices.shape();
    std::vector<std::size_t> new_shape = {ishape[0], n_classes};

    xt::xarray<float> one_hot_labels(new_shape);
    one_hot_labels.fill(0.0f);

    // coudl not make it work with xt::index_view
    for (std::size_t i = 0; i < indices.size(); ++i) {
        // Convert indices(i) to std::size_t to index properly
        std::size_t class_index = static_cast<std::size_t>(indices(i));
        one_hot_labels(i, class_index) = 1.0f;
    }

    return one_hot_labels;
}

xt::xarray<float> cross_entropy_forward(const xt::xarray<float>& logits, const xt::xarray<int>& targets) {
    size_t N = logits.shape()[0];
    size_t S = logits.shape()[2];
    size_t C = logits.shape()[3];
    auto logits_2d = logits;
    logits_2d.reshape({N * S, C});
    auto targets_1d = targets;
    targets_1d.reshape({N * S});
    xt::xarray<float> log_probs = log_softmax(logits_2d);
    std::cout << "log_probs shape: " << xt::adapt(log_probs.shape()) << std::endl;
    xt::xarray<float> one_hot_targets = one_hot(logits.shape()[3], targets_1d);
    std::cout << "one_hot_targets shape: " << xt::adapt(one_hot_targets.shape()) << std::endl;

    xt::xarray<float> loss = -xt::sum(one_hot_targets * log_probs) / logits_2d.shape()[0];
    std::cout << loss[0] << std::endl;
    return loss;
}

xt::xarray<float> cross_entropy_backward(const xt::xarray<float>& logits, const xt::xarray<int>& targets) {
    size_t N = logits.shape()[0];
    size_t S = logits.shape()[2];
    size_t C = logits.shape()[3];
    auto logits_2d = logits;
    logits_2d.reshape({N * S, C});
    auto targets_1d = targets;
    targets_1d.reshape({N * S});
    std::cout << "targets_1d shape: " << xt::adapt(targets_1d.shape()) << std::endl;
    std::cout << "logits_2d shape: " << xt::adapt(logits_2d.shape()) << std::endl;
    // Compute softmax
    auto probs = softmax(logits_2d);  // [1,1,N, C]

    // grad = (probs - one_hot(targets)) / N
    // We can do this in-place
    for (size_t i = 0; i < N * S; ++i) {
        probs(i, targets_1d(i)) -= 1.0;
    }
    probs /= static_cast<float>(N * S);
    probs.reshape({N, 1, S, C});

    return probs;
}

autograd::TensorPtr cross_entropy_loss_xtensor(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target, ReduceType reduce) {
    auto pred_xtensor = core::to_xtensor(prediction->get_value());
    auto target_xtensor = core::to_xtensor<int32_t>(target->get_value());

    auto loss = cross_entropy_forward(pred_xtensor, target_xtensor);

    auto out = autograd::create_tensor(core::from_xtensor(loss, &autograd::ctx().get_device()));

    autograd::GradFunction grad = [out, prediction, pred_xtensor, target_xtensor]() {
        auto gradx = cross_entropy_backward(pred_xtensor, target_xtensor);
        auto grad = core::from_xtensor(gradx, &autograd::ctx().get_device());
        grad = ttnn::multiply(grad, out->get_grad());
        prediction->add_grad(grad);
    };

    auto links = autograd::get_links(prediction);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr cross_entropy_loss(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target, ReduceType reduce) {
    auto loss = cross_entropy_loss_without_reduce_(prediction, target);
    if (reduce == ReduceType::MEAN) {
        return ops::mean(loss);
    } else {
        throw std::logic_error("Unsupported cross entropy reduction type");
    }
}

autograd::TensorPtr nll_loss(
    const autograd::TensorPtr& prediction, const autograd::TensorPtr& target, ReduceType reduce) {
    if (reduce != ReduceType::MEAN) {
        throw std::logic_error("Unsupported NLL reduction type, only MEAN is supported");
    }

    auto* device = &autograd::ctx().get_device();
    auto divisor = core::empty(ttnn::Shape({1, 1}, {32, 32}), device, prediction->get_value().memory_config());

    auto tensor_shape = prediction->get_value().shape();
    uint32_t Ndim = tensor_shape[0] * tensor_shape[1] * tensor_shape[2];
    uint32_t Cdim = tensor_shape[3];
    auto reshaped_tensor = ttnn::reshape(prediction->get_value(), ttnn::Shape({Ndim, Cdim}));
    auto loss_tensor = ttnn::moreh_nll_loss(
        reshaped_tensor,
        target->get_value(),
        /* reduction */ "mean",
        /* weight_tensor */ std::nullopt,
        /* divisor_tensor */ divisor,
        /* output_tensor */ std::nullopt,
        /* ignore_index */ -100,
        /* memory_config */ prediction->get_value().memory_config(),
        /* compute_kernel_config */ core::ComputeKernelConfig::precise());
    auto out = autograd::create_tensor(loss_tensor);

    autograd::GradFunction grad = [prediction, target, out, Ndim, Cdim, device, divisor]() {
        auto out_grad = core::empty(ttnn::Shape({Ndim, Cdim}), device, prediction->get_value().memory_config());
        auto grad = ttnn::moreh_nll_loss_backward(
            target->get_value(),
            out->get_grad(),
            /* reduction_mean */ true,
            /* weight_tensor */ std::nullopt,
            /* input_grad_tensor */ out_grad,
            /* divisor_tensor */ divisor,
            /* ignore_index */ -100,
            /* memory_config */ std::nullopt,
            /* compute_kernel_config */ core::ComputeKernelConfig::precise());
        grad = ttnn::reshape(grad, prediction->get_value().shape());
        prediction->add_grad(grad);
    };
    auto links = autograd::get_links(prediction);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
