// SPDX-FileCopyrightText: 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <utility>

#include <tt-metalium/buffer_constants.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"

namespace ttnn {
namespace operations::conv {
using namespace tt;
using sliding_window::ParallelConfig;
using sliding_window::SlidingWindowConfig;

namespace conv2d {

using OutputHeight = uint32_t;
using OutputWidth = uint32_t;
using Result = std::tuple<ttnn::Tensor, OutputHeight, OutputWidth, ttnn::Tensor, std::optional<ttnn::Tensor>>;

template <typename T>
Result conv2d(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    T* device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config) {
    // Debug prints
    std::cout << "KCM CPP ttnn::conv2d parameters: " << std::endl;
    std::cout << "input_tensor shape=[" << input_tensor.get_logical_shape()[0] << ", "
              << input_tensor.get_logical_shape()[1] << ", " << input_tensor.get_logical_shape()[2] << ", "
              << input_tensor.get_logical_shape()[3] << "]" << std::endl;

    std::cout << "weight_tensor shape=[" << weight_tensor.get_logical_shape()[0] << ", "
              << weight_tensor.get_logical_shape()[1] << ", " << weight_tensor.get_logical_shape()[2] << ", "
              << weight_tensor.get_logical_shape()[3] << "]" << std::endl;

    std::cout << "in_channels=" << in_channels << ", out_channels=" << out_channels << std::endl;
    std::cout << "device=" << device << std::endl;

    std::cout << "bias_tensor=";
    if (bias_tensor.has_value()) {
        std::cout << "[";
        const auto& bias_shape = bias_tensor.value().get_logical_shape();
        for (size_t i = 0; i < bias_shape.size(); ++i) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << bias_shape[i];
        }
        std::cout << "]";
    } else {
        std::cout << "None";
    }
    std::cout << std::endl;

    std::cout << "kernel_size=[" << kernel_size[0] << ", " << kernel_size[1] << "]" << std::endl;
    std::cout << "stride=[" << stride[0] << ", " << stride[1] << "]" << std::endl;

    // Handle different types of padding
    std::cout << "padding=";
    std::visit(
        [](auto&& arg) {
            using ArgType = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<ArgType, std::array<uint32_t, 2>>) {
                std::cout << "[" << arg[0] << ", " << arg[1] << "]";
            } else if constexpr (std::is_same_v<ArgType, std::array<uint32_t, 4>>) {
                std::cout << "[" << arg[0] << ", " << arg[1] << ", " << arg[2] << ", " << arg[3] << "]";
            }
        },
        padding);
    std::cout << std::endl;

    std::cout << "dilation=[" << dilation[0] << ", " << dilation[1] << "]" << std::endl;
    std::cout << "groups=" << groups << std::endl;
    std::cout << "batch_size=" << batch_size << std::endl;
    std::cout << "input_height=" << input_height << std::endl;
    std::cout << "input_width=" << input_width << std::endl;

    // Print all Conv2dConfig fields
    std::cout << "Conv2dConfig=" << std::endl;
    std::cout << "{" << std::endl;

    // Print data types
    Conv2dConfig config = conv_config_.value_or(Conv2dConfig{});
    std::cout << "  dtype=";
    switch (config.dtype) {
        case tt::tt_metal::DataType::BFLOAT16: std::cout << "BFLOAT16"; break;
        case tt::tt_metal::DataType::BFLOAT8_B: std::cout << "BFLOAT8_B"; break;
        case tt::tt_metal::DataType::FLOAT32: std::cout << "FLOAT32"; break;
        default: std::cout << "OTHER"; break;
    }
    std::cout << std::endl;

    std::cout << "  weights_dtype=";
    switch (config.weights_dtype) {
        case tt::tt_metal::DataType::BFLOAT16: std::cout << "BFLOAT16"; break;
        case tt::tt_metal::DataType::BFLOAT8_B: std::cout << "BFLOAT8_B"; break;
        case tt::tt_metal::DataType::FLOAT32: std::cout << "FLOAT32"; break;
        default: std::cout << "OTHER"; break;
    }
    std::cout << std::endl;

    std::cout << "  input_channels_alignment=" << config.input_channels_alignment << std::endl;
    std::cout << "  deallocate_activation=" << (config.deallocate_activation ? "true" : "false") << std::endl;
    std::cout << "  reallocate_halo_output=" << (config.reallocate_halo_output ? "true" : "false") << std::endl;
    std::cout << "  act_block_h_override=" << config.act_block_h_override << std::endl;
    std::cout << "  act_block_w_div=" << config.act_block_w_div << std::endl;
    std::cout << "  reshard_if_not_optimal=" << (config.reshard_if_not_optimal ? "true" : "false") << std::endl;
    std::cout << "  override_sharding_config=" << (config.override_sharding_config ? "true" : "false") << std::endl;

    // Print core_grid if available
    std::cout << "  core_grid=";
    if (config.core_grid.has_value()) {
        std::cout << "[CORE_GRID_DEFINED]";  // Simplified representation
    } else {
        std::cout << "None";
    }
    std::cout << std::endl;

    // Print output_layout enum
    std::cout << "  output_layout=";
    switch (config.output_layout) {
        case Layout::ROW_MAJOR: std::cout << "ROW_MAJOR"; break;
        case Layout::TILE: std::cout << "TILE"; break;
        case Layout::INVALID: std::cout << "INVALID"; break;
        default: std::cout << "UNKNOWN"; break;
    }
    std::cout << std::endl;

    std::cout << "  activation=\"" << config.activation << "\"" << std::endl;
    std::cout << "  transpose_shards=" << (config.transpose_shards ? "true" : "false") << std::endl;
    std::cout << "  preprocess_weights_on_device=" << (config.preprocess_weights_on_device ? "true" : "false")
              << std::endl;
    std::cout << "  always_preprocess_weights=" << (config.always_preprocess_weights ? "true" : "false") << std::endl;
    std::cout << "  enable_act_double_buffer=" << (config.enable_act_double_buffer ? "true" : "false") << std::endl;
    std::cout << "  enable_weights_double_buffer=" << (config.enable_weights_double_buffer ? "true" : "false")
              << std::endl;
    std::cout << "  enable_split_reader=" << (config.enable_split_reader ? "true" : "false") << std::endl;
    std::cout << "  enable_subblock_padding=" << (config.enable_subblock_padding ? "true" : "false") << std::endl;
    std::cout << "  in_place=" << (config.in_place ? "true" : "false") << std::endl;

    std::cout << "  shard_layout=";
    if (config.shard_layout.has_value()) {
        auto shard_layout = config.shard_layout.value();
        // Convert the TensorMemoryLayout enum to string instead of trying to stream it directly
        switch (shard_layout) {
            case TensorMemoryLayout::INTERLEAVED: std::cout << "INTERLEAVED"; break;
            case TensorMemoryLayout::SINGLE_BANK: std::cout << "SINGLE_BANK"; break;
            case TensorMemoryLayout::HEIGHT_SHARDED: std::cout << "HEIGHT_SHARDED"; break;
            case TensorMemoryLayout::WIDTH_SHARDED: std::cout << "WIDTH_SHARDED"; break;
            case TensorMemoryLayout::BLOCK_SHARDED: std::cout << "BLOCK_SHARDED"; break;
            default: std::cout << "UNKNOWN"; break;
        }
    } else {
        std::cout << "None";
    }
    std::cout << std::endl;

    std::cout << "}" << std::endl;

    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    std::array<uint32_t, 4> padding_n4 = sliding_window::get_pair_n4_padding(padding);
    const bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding_n4, dilation, groups, conv_config);
    auto [output_height, output_width] =
        calculate_output_image_size({input_height, input_width}, kernel_size, stride, padding_n4, dilation);


    DeviceComputeKernelConfig compute_config = compute_config_.value_or(get_conv_default_compute_kernel_config(device));

    const auto compute_grid_size = device->compute_with_storage_grid_size();

    bool auto_shard = false;
    if (!input_tensor.is_sharded() && !conv_config.shard_layout.has_value()) {
        // In this case we deduce the shard layout.
        conv_config = determine_conv_config_for_auto_shard(
            conv_config,
            mm_conv,
            batch_size,
            in_channels,
            out_channels,
            output_height,
            output_width,
            weight_tensor.get_logical_shape()[3],
            input_height,
            input_width,
            compute_grid_size,
            input_tensor.layout(),
            ttnn::is_tensor_on_device_or_multidevice(input_tensor) ? std::make_optional(input_tensor.memory_config())
                                                                   : std::nullopt,
            kernel_size,
            groups,
            bias_tensor.has_value(),
            compute_config);
        auto_shard = true;
    }

    ShardOrientation shard_orientation =
        conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;

    auto [input_tensor_post_tm, parallel_config, output_parallel_config] = shard_or_reshard_tensor_if_required(
        device,
        input_tensor,
        conv_config,
        batch_size,
        output_height,
        output_width,
        in_channels,
        out_channels,
        mm_conv,
        auto_shard);

    auto [opt_conv_op_parallel_config, opt_conv_op_block_config, conv_out_memory_config] = get_conv_configs(
        conv_config,
        compute_config,
        parallel_config,
        output_parallel_config,
        in_channels,
        out_channels,
        batch_size,
        output_height,
        output_width,
        kernel_size,
        compute_grid_size);

    bool weight_is_on_device = ttnn::is_tensor_on_device_or_multidevice(weight_tensor);
    ttnn::Tensor weight_tensor_on_device = weight_tensor;
    std::optional<ttnn::Tensor> bias_tensor_on_device = bias_tensor;
    if (!weight_is_on_device || conv_config.always_preprocess_weights) {
        // prepare weights in desired layout and move to device

        // TODO: Implement heuristic to decide if weights should be preprocessed on device.
        if (conv_config.preprocess_weights_on_device == false) {
            tie(weight_tensor_on_device, bias_tensor_on_device) = prepare_conv_weights_biases_and_move_to_device(
                weight_tensor,
                bias_tensor,
                conv_config.input_channels_alignment,
                conv_config.weights_dtype,
                opt_conv_op_block_config.act_block_w_ntiles,
                opt_conv_op_block_config.out_subblock_w_ntiles,
                parallel_config,
                output_parallel_config,
                device,
                groups,
                opt_conv_op_block_config.act_block_h_ntiles,
                input_width,
                true);
        } else {
            tie(weight_tensor_on_device, bias_tensor_on_device) = prepare_conv_weights_biases_on_device(
                weight_tensor,
                bias_tensor,
                conv_config.input_channels_alignment,
                conv_config.weights_dtype,
                opt_conv_op_block_config.act_block_w_ntiles,
                opt_conv_op_block_config.out_subblock_w_ntiles,
                parallel_config,
                output_parallel_config,
                device,
                groups,
                opt_conv_op_block_config.act_block_h_ntiles,
                input_width,
                true);
        }
    }
    // if 1x1 conv w/ stride 1, convert input tensor to tile layout if required
    if (mm_conv) {
        input_tensor_post_tm = ttnn::to_layout(
            input_tensor_post_tm, Layout::TILE, conv_config.dtype, input_tensor_post_tm.memory_config(), device);
    }
    // call optimized conv op or matmul micro op
    bool input_is_on_device = ttnn::is_tensor_on_device_or_multidevice(input_tensor_post_tm);
    TT_ASSERT(input_is_on_device);

    if (!mm_conv) {
        // call halo op
        SlidingWindowConfig sliding_window_config = SlidingWindowConfig{
            .batch_size = batch_size,
            .input_hw = {input_height, input_width},
            .window_hw = {kernel_size[0], kernel_size[1]},
            .stride_hw = {stride[0], stride[1]},
            .padding = {{padding_n4[0], padding_n4[1], padding_n4[2], padding_n4[3]}},
            .dilation_hw = {dilation[0], dilation[1]},
            .num_cores_nhw = opt_conv_op_parallel_config.num_cores_nhw,
            .core_range_set = input_tensor_post_tm.memory_config().shard_spec.value().grid,
            .snap_to_tile = true,
        };

        bool bypass_halo =
            (parallel_config.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED &&
             sliding_window_config.get_pad_h() == 0 && sliding_window_config.get_pad_w() == 0);

        if (bypass_halo) {
            if (input_tensor_post_tm.layout() == Layout::TILE) {
                // Reshape is used as a workaround to an issue in to_layout mentioned here :
                // https://github.com/tenstorrent/tt-metal/issues/16330
                input_tensor_post_tm = ttnn::reshape(input_tensor_post_tm, input_tensor_post_tm.get_padded_shape());
                input_tensor_post_tm =
                    ttnn::to_layout(input_tensor_post_tm, Layout::ROW_MAJOR, std::nullopt, std::nullopt, device);
            }
        } else {
            Tensor halo_output = ttnn::halo(
                DefaultQueueId,
                input_tensor_post_tm,
                sliding_window_config,
                0,
                false,
                parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
                0,
                input_tensor_post_tm.memory_config(),
                true,
                conv_config.in_place);

            if (conv_config.deallocate_activation) {
                input_tensor_post_tm.deallocate(/*force*/ true);
            }

            input_tensor_post_tm = std::move(halo_output);

            if (conv_config.reallocate_halo_output) {
                input_tensor_post_tm = ttnn::move(input_tensor_post_tm);
            }
        }

        // call conv micro op
        auto conv_output = optimized_conv_new(
            input_tensor_post_tm,
            weight_tensor_on_device,
            bias_tensor_on_device,
            sliding_window_config,
            out_channels,
            groups,
            conv_config.output_layout == Layout::ROW_MAJOR,
            conv_config.activation,
            opt_conv_op_parallel_config,
            opt_conv_op_block_config,
            conv_out_memory_config,
            conv_config.dtype,
            {batch_size, input_height, input_width, in_channels},
            conv_config.input_channels_alignment == 16,
            compute_config,
            conv_config.enable_act_double_buffer,
            conv_config.enable_weights_double_buffer,
            conv_config.enable_split_reader);

        if (memory_config.has_value() && memory_config.value() != conv_output.memory_config()) {
            conv_output = ttnn::to_memory_config(conv_output, memory_config.value(), std::nullopt);
        }
        return {conv_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
    } else {
        // run conv as matmul
        std::optional<ttnn::operations::matmul::MatmulProgramConfig> program_config = std::nullopt;
        std::optional<MemoryConfig> mm_output_memory_config = std::nullopt;
        std::optional<std::string> activation = std::nullopt;

        if (input_tensor_post_tm.is_sharded()) {
            uint32_t num_cores_c = get_num_cores_channels_from_parallel_config(parallel_config);
            program_config = determine_matmul_op_config_from_conv_op_config(
                opt_conv_op_parallel_config,
                opt_conv_op_block_config,
                parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED,
                conv_config.activation,
                parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
                num_cores_c);
            mm_output_memory_config = conv_out_memory_config;
        } else {
            if (!conv_config.activation.empty()) {
                activation = conv_config.activation;
            }
        }
        Tensor matmul_output = ttnn::linear(
            input_tensor_post_tm,
            weight_tensor_on_device,
            bias_tensor_on_device,
            false,
            false,
            mm_output_memory_config,
            std::nullopt,
            program_config,
            activation);

        if (memory_config.has_value() && memory_config.value() != matmul_output.memory_config()) {
            matmul_output = ttnn::to_memory_config(matmul_output, memory_config.value(), std::nullopt);
        }

        return {matmul_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
    }
}

Result Conv2dOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    IDevice* device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config) {
    // Debug prints for IDevice overload
    // std::cout << "[IDevice] ttnn::conv2d::invoke parameters: ";
    // std::cout << "input_tensor shape=[" << input_tensor.get_logical_shape()[0] << ", "
    //           << input_tensor.get_logical_shape()[1] << ", " << input_tensor.get_logical_shape()[2] << ", "
    //           << input_tensor.get_logical_shape()[3] << "], ";

    // std::cout << "weight_tensor shape=[" << weight_tensor.get_logical_shape()[0] << ", "
    //           << weight_tensor.get_logical_shape()[1] << ", " << weight_tensor.get_logical_shape()[2] << ", "
    //           << weight_tensor.get_logical_shape()[3] << "], ";

    // std::cout << "in_channels=" << in_channels << ", out_channels=" << out_channels << ", ";
    // std::cout << "batch_size=" << batch_size << ", ";
    // std::cout << "input_height=" << input_height << ", input_width=" << input_width << ", ";
    // std::cout << "kernel_size=(" << kernel_size[0] << ", " << kernel_size[1] << "), ";
    // std::cout << "stride=(" << stride[0] << ", " << stride[1] << "), ";
    // std::cout << "groups=" << groups << std::endl;

    return conv2d(
        input_tensor,
        weight_tensor,
        device,
        in_channels,
        out_channels,
        batch_size,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        std::move(bias_tensor),
        std::move(conv_config_),
        std::move(compute_config_),
        memory_config);
}

Result Conv2dOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    MeshDevice* device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config) {
    // Debug prints for MeshDevice overload
    // std::cout << "[MeshDevice] ttnn::conv2d::invoke parameters: ";
    // std::cout << "input_tensor shape=[" << input_tensor.get_logical_shape()[0] << ", "
    //           << input_tensor.get_logical_shape()[1] << ", " << input_tensor.get_logical_shape()[2] << ", "
    //           << input_tensor.get_logical_shape()[3] << "], ";

    // std::cout << "weight_tensor shape=[" << weight_tensor.get_logical_shape()[0] << ", "
    //           << weight_tensor.get_logical_shape()[1] << ", " << weight_tensor.get_logical_shape()[2] << ", "
    //           << weight_tensor.get_logical_shape()[3] << "], ";

    // std::cout << "in_channels=" << in_channels << ", out_channels=" << out_channels << ", ";
    // std::cout << "batch_size=" << batch_size << ", ";
    // std::cout << "input_height=" << input_height << ", input_width=" << input_width << ", ";
    // std::cout << "kernel_size=(" << kernel_size[0] << ", " << kernel_size[1] << "), ";
    // std::cout << "stride=(" << stride[0] << ", " << stride[1] << "), ";
    // std::cout << "groups=" << groups << std::endl;

    return conv2d(
        input_tensor,
        weight_tensor,
        device,
        in_channels,
        out_channels,
        batch_size,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        std::move(bias_tensor),
        std::move(conv_config_),
        std::move(compute_config_),
        memory_config);
}

}  // namespace conv2d
}  // namespace operations::conv
}  // namespace ttnn
