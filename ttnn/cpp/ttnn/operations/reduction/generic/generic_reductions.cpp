// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/core/core.hpp"

namespace {

// Some tensors are pre-padded with 0s. E.g. Those generated via from_torch.
// Therefore need to always pad tensors again. To do that, convert to row major,
// pad, and then convert back to tile layout.
// Limitations of pad require transpose, un-transpose, and then slicing to isolate values of interest.
// End result will be padded, and after reduce is done, will need to be sliced back.
ttnn::Tensor pad_tensor_with_value(const ttnn::Tensor& input_tensor, float pad_value) {
    ttnn::Shape with_padding = input_tensor.get_shape().with_tile_padding();
    ttnn::Tensor intermediate_tensor =
        ttnn::to_layout(input_tensor, Layout::ROW_MAJOR, std::nullopt, std::nullopt, input_tensor.device());
    intermediate_tensor = ttnn::transpose(intermediate_tensor, 0, 3);
    // use transposed values for padded_shape
    tt::tt_metal::Array4D padded_shape = {with_padding[3], with_padding[1], with_padding[2], with_padding[0]};
    ttnn::Tensor padded_tensor =
        ttnn::pad(intermediate_tensor, padded_shape, tt::tt_metal::Array4D({0, 0, 0, 0}), pad_value);
    padded_tensor = ttnn::transpose(padded_tensor, 0, 3);
    std::array<uint32_t, 4> begins = {0, 0, 0, 0};
    std::array<uint32_t, 4> ends = {with_padding[0], with_padding[1], with_padding[2], with_padding[3]};
    std::array<uint32_t, 4> step = {1, 1, 1, 1};
    padded_tensor = ttnn::slice(padded_tensor, begins, ends, step);
    padded_tensor = ttnn::to_layout(padded_tensor, Layout::TILE, std::nullopt, std::nullopt, padded_tensor.device());
    tt::log_debug(tt::LogOp, "max {} {} {}", padded_shape, pad_value, padded_tensor);
    return padded_tensor;
}

// Pad tensor with values, reduce, and then slice back to un-padded size.
ttnn::Tensor reduce_with_padding(
    ttnn::Tensor& input_tensor,
    float pad_value,
    tt::tt_metal::ReduceOpMath op,
    const tt::tt_metal::ReduceOpDim reduce_op_dim,
    float scalar,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    ttnn::Tensor padded_tensor = pad_tensor_with_value(input_tensor, pad_value);
    ttnn::Tensor output_tensor = tt::tt_metal::reduce(
        padded_tensor, op, reduce_op_dim, scalar, memory_config, std::nullopt, compute_kernel_config);
    ttnn::Shape shape = input_tensor.get_shape();
    std::array<uint32_t, 4> begins = {0, 0, 0, 0};
    std::array<uint32_t, 4> ends = {shape[0], shape[1], shape[2], shape[3]};
    std::array<uint32_t, 4> step = {1, 1, 1, 1};
    if (reduce_op_dim == tt::tt_metal::ReduceOpDim::W) {
        ends[3] = 1;
    } else if (reduce_op_dim == tt::tt_metal::ReduceOpDim::H) {
        ends[2] = 1;
    } else if (reduce_op_dim == tt::tt_metal::ReduceOpDim::HW) {
        ends[2] = 1;
        ends[3] = 1;
    } else {
        TT_THROW("Unsupported reduce op dim {}", reduce_op_dim);
    }

    output_tensor = ttnn::slice(output_tensor, begins, ends, step);
    return output_tensor;
}

ttnn::Tensor tree_add_tensors(std::vector<ttnn::Tensor> input_tensors, uint32_t start, uint32_t length) {
    tt::log_debug(tt::LogOp, "tree_add_tensors {} {} {}", input_tensors.size(), start, length);
    TT_FATAL(length > 0, "length for tree_add_tensors cannot be 0");
    if (length == 1) {
        return input_tensors[start];
    } else if (length == 2) {
        return ttnn::add(input_tensors[start], input_tensors[start + 1]);
    }
    int offset = length / 2;
    ttnn::Tensor tensor0 = tree_add_tensors(input_tensors, start, offset);
    ttnn::Tensor tensor1 = tree_add_tensors(input_tensors, start + offset, length - offset);
    return ttnn::add(tensor0, tensor1);
}

constexpr uint32_t REDUCE_TREE_ADD_MIN_SIZE = 512;  // Must be a multiple of tile size

// Divide tensor into evenly sized sub-tensors.
// All entries need to start at tile boundary, which may require adjustment of last sub-tensor.
std::vector<ttnn::Tensor> get_subdivided_tensors(
    const ttnn::Tensor& input_tensor, const ttnn::Shape& shape, uint32_t dimension, uint32_t count) {
    size_t rank = shape.rank();
    std::vector<ttnn::Tensor> intermediate_tensors;
    std::array<uint32_t, 4> begins = {0, 0, 0, 0};
    std::array<uint32_t, 4> ends = {shape[0], shape[1], shape[2], shape[3]};
    std::array<uint32_t, 4> step = {1, 1, 1, 1};
    ends[dimension] = REDUCE_TREE_ADD_MIN_SIZE;
    uint32_t length = shape[dimension];
    for (uint32_t i = 0; i < count - 1; i++) {
        intermediate_tensors.push_back(ttnn::slice(input_tensor, begins, ends, step));
        begins[dimension] += REDUCE_TREE_ADD_MIN_SIZE;
        ends[dimension] += REDUCE_TREE_ADD_MIN_SIZE;
    }
    if (length == count * REDUCE_TREE_ADD_MIN_SIZE) {
        intermediate_tensors.push_back(ttnn::slice(input_tensor, begins, ends, step));
    } else {
        // Last entry goes up to the end of the tensor.
        // Then needs to be aligned to logical shape and padded.
        // Padding needs to be for last two dimensions, which means that the tensor needs
        // to then also be sliced back since only one dimension should be extended.
        ends[dimension] = length;
        ttnn::Tensor intermediate_tensor = ttnn::slice(input_tensor, begins, ends, step);
        ttnn::Shape with_padding = shape.with_tile_padding();
        tt::tt_metal::Array4D padded_shape = {with_padding[0], with_padding[1], with_padding[2], with_padding[3]};
        padded_shape[dimension] = REDUCE_TREE_ADD_MIN_SIZE;
        tt::tt_metal::Array4D pad_zeros = {0, 0, 0, 0};
        intermediate_tensor = ttnn::pad(0, intermediate_tensor, padded_shape, pad_zeros, 0.0, true, std::nullopt);
        begins[dimension] = 0;
        ends[dimension] = REDUCE_TREE_ADD_MIN_SIZE;
        intermediate_tensor = ttnn::slice(intermediate_tensor, begins, ends, step);
        intermediate_tensors.push_back(intermediate_tensor);
    }
    return intermediate_tensors;
}

ttnn::Tensor reduce_sum_by_tree_add(
    const ttnn::Tensor& input_tensor,
    const tt::tt_metal::ReduceOpDim reduce_op_dim,
    const ttnn::MemoryConfig& output_mem_config,
    const std::optional<ttnn::DataType>& output_dtype,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(
        reduce_op_dim == tt::tt_metal::ReduceOpDim::W || reduce_op_dim == tt::tt_metal::ReduceOpDim::H,
        "Only W and H reduce op dim supported for reduce sum by tree add. Instead {} passed in",
        reduce_op_dim);
    const ttnn::Shape& input_shape = input_tensor.get_shape();
    size_t rank = input_shape.rank();
    uint32_t dimension = reduce_op_dim == tt::tt_metal::ReduceOpDim::W ? (rank - 1) : (rank - 2);
    uint32_t length = input_shape[dimension];
    uint32_t count = tt::div_up(length, REDUCE_TREE_ADD_MIN_SIZE);
    const ttnn::Tensor& intermediate_tensor =
        count == 1 ? input_tensor
                   : tree_add_tensors(
                         std::move(get_subdivided_tensors(input_tensor, input_shape, dimension, count)), 0, count);
    return tt::tt_metal::reduce(
        intermediate_tensor,
        tt::tt_metal::ReduceOpMath::SUM,
        reduce_op_dim,
        1.0,
        output_mem_config,
        std::nullopt,
        compute_kernel_config);
}

ttnn::Tensor reduce_sum(
    const ttnn::Tensor& input_tensor,
    const tt::tt_metal::ReduceOpDim reduce_op_dim,
    float scalar,
    const ttnn::MemoryConfig& output_mem_config,
    const std::optional<ttnn::DataType>& output_dtype,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    ttnn::Tensor output_tensor;
    tt::log_debug(tt::LogOp, "input {}", input_tensor);
    input_tensor.print();
    if (reduce_op_dim == tt::tt_metal::ReduceOpDim::W || reduce_op_dim == tt::tt_metal::ReduceOpDim::H) {
        output_tensor =
            reduce_sum_by_tree_add(input_tensor, reduce_op_dim, output_mem_config, std::nullopt, compute_kernel_config);
    } else {
        output_tensor = tt::tt_metal::reduce(
            input_tensor,
            tt::tt_metal::ReduceOpMath::SUM,
            reduce_op_dim,
            1.0,
            output_mem_config,
            std::nullopt,
            compute_kernel_config);
    }
    tt::log_debug(tt::LogOp, "before multiply {}", output_tensor);
    output_tensor.print();
    ttnn::multiply_(output_tensor, scalar);
    tt::log_debug(tt::LogOp, "after multiply {}", output_tensor);
    output_tensor.print();
    return output_tensor;
}

}  // namespace

namespace ttnn {
namespace operations::reduction {

template <ReduceType reduce_type>
static Tensor reduce_impl(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, ttnn::SmallVector<int>>>& dim_arg,
    const bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    bool reshape) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    auto input_shape = input_tensor_arg.get_shape();
    auto rank = input_shape.size();
    auto memory_config = memory_config_arg.value_or(input_tensor_arg.memory_config());

    ttnn::SmallVector<int> dim{};
    if (dim_arg.has_value()) {
        if (not std::holds_alternative<ttnn::SmallVector<int>>(dim_arg.value())) {
            auto dim_as_int = std::get<int>(dim_arg.value());
            dim = ttnn::SmallVector<int>({dim_as_int});
        } else {
            dim = std::get<ttnn::SmallVector<int>>(dim_arg.value());
        }
    } else {
        dim = ttnn::SmallVector<int>(rank);
        for (int i = 0; i < rank; i++) {
            dim[i] = i;
        }
    }

    for (int i = 0; i < dim.size(); i++) {
        if (dim[i] < 0) {
            dim[i] += rank;
        }
        int dim_i = dim[i];
        TT_FATAL(
            dim_i >= 0 && dim_i < rank,
            "Unsupported dim {} at index {}. After possible adjustment, needs to be at least 0 and less than rank {}",
            dim_i,
            i,
            rank);
    }

    if (dim.size() == 1 && rank == 4) {
        if (dim[0] == rank - 3) {
            // Pad before running the op to only pay cost of formatting once
            // auto input_tensor_pad_shape = AutoFormat::pad_to_tile_shape(input_tensor_arg.get_legacy_shape(), true);
            auto out_shape = input_tensor_arg.get_legacy_shape();
            out_shape[1] = 1;

            // auto formatted_input_tensor = input_tensor_arg;
            // float pad_value = (reduce_type == ReduceType::Max)   ? -std::numeric_limits<float>::infinity()
            //                   : (reduce_type == ReduceType::Min) ? std::numeric_limits<float>::infinity()
            //                                                      : 0;

            // if (!AutoFormat::check_input_tensor_format(input_tensor_arg, input_tensor_pad_shape)) {
            //     formatted_input_tensor = AutoFormat::format_input_tensor(
            //         input_tensor_arg, input_tensor_arg.device(), input_tensor_pad_shape, pad_value, Layout::TILE);
            // }
            // Tensor output = ttnn::transpose(formatted_input_tensor, 1, -2, memory_config);
            Tensor output = ttnn::transpose(input_tensor_arg, 1, -2, memory_config);
            output = reduce_impl<reduce_type>(output, 2, keepdim, memory_config, compute_kernel_config, scalar, false);
            output = ttnn::transpose(output, 1, -2, memory_config);
            return AutoFormat::format_output_tensor(output, out_shape, input_tensor_arg.device(), Layout::TILE);
        } else if (dim[0] == 0) {
            // Pad before running the op to only pay cost of formatting once
            // auto input_tensor_pad_shape =
            //    AutoFormat::pad_to_tile_shape(input_tensor_arg.get_legacy_shape(), false, true);
            auto out_shape = input_tensor_arg.get_legacy_shape();
            out_shape[0] = 1;

            // auto formatted_input_tensor = input_tensor_arg;
            // if (!AutoFormat::check_input_tensor_format(input_tensor_arg, input_tensor_pad_shape)) {
            //     formatted_input_tensor = AutoFormat::format_input_tensor(
            //         input_tensor_arg, input_tensor_arg.device(), input_tensor_pad_shape, 0.0, Layout::TILE);
            // }
            // Tensor output = ttnn::transpose(formatted_input_tensor, 0, -2, memory_config);
            Tensor output = ttnn::transpose(input_tensor_arg, 0, -2, memory_config);
            output = reduce_impl<reduce_type>(output, 2, keepdim, memory_config, compute_kernel_config, scalar, false);
            output = ttnn::transpose(output, 0, -2, memory_config);
            return AutoFormat::format_output_tensor(output, out_shape, input_tensor_arg.device(), Layout::TILE);
        }
    }

    std::sort(dim.begin(), dim.end());

    ttnn::SmallVector<uint32_t> output_shape;
    ttnn::SmallVector<uint32_t> padded_output_shape;
    for (int axis = 0; axis < input_shape.size(); axis++) {
        if (std::find(dim.begin(), dim.end(), axis) != dim.end()) {
            if (keepdim) {
                output_shape.push_back(1);
                padded_output_shape.push_back(axis >= rank - 2 ? ttnn::TILE_SIZE : 1);
            }
        } else {
            // Get the shape for the output tensor
            output_shape.push_back(input_shape[axis]);
            // Get the padded shape for the output tensor
            padded_output_shape.push_back(input_shape.value[axis]);
        }
        tt::log_debug(
            tt::LogOp,
            "axis {} input {} {} output {} {}",
            axis,
            input_shape[axis],
            input_shape.value[axis],
            output_shape,
            padded_output_shape);
    }

    auto input_tensor = ttnn::unsqueeze_to_4D(input_tensor_arg);

    Tensor output_tensor;
    if (!dim_arg.has_value()) {
        if constexpr (
            reduce_type == ReduceType::Sum || reduce_type == ReduceType::Max || reduce_type == ReduceType::Min) {
            output_tensor = input_tensor;
            for (int rank = input_tensor.get_legacy_shape().rank() - 1; rank >= 0; rank--) {
                output_tensor = reduce_impl<reduce_type>(
                    output_tensor, rank, true, memory_config, compute_kernel_config, scalar, false);
            }
        } else if constexpr (reduce_type == ReduceType::Mean) {
            output_tensor = input_tensor;
            for (int rank = input_tensor.get_legacy_shape().rank() - 1; rank >= 0; rank--) {
                output_tensor = reduce_impl<ReduceType::Sum>(
                    output_tensor, rank, true, memory_config, compute_kernel_config, scalar, false);
            }
            float inv_volume = 1.0f / input_tensor.get_logical_volume();
            output_tensor = ttnn::mul_sfpu(inv_volume, output_tensor, memory_config);
        } else {
            TT_THROW("Unsupported reduction operation");
        }
    } else {
        tt::tt_metal::ReduceOpDim reduce_op_dim;
        if (dim.size() == 1 and dim[0] == rank - 1) {
            reduce_op_dim = tt::tt_metal::ReduceOpDim::W;
        } else if (dim.size() == 1 and dim[0] == rank - 2) {
            reduce_op_dim = tt::tt_metal::ReduceOpDim::H;
        } else if (dim.size() == 2 and dim[0] == rank - 2 and dim[1] == rank - 1) {
            reduce_op_dim = tt::tt_metal::ReduceOpDim::HW;
        } else {
            TT_THROW("Unsupported dim");
        }

        int reduced_volume = 1;
        for (int axis : dim) {
            reduced_volume *= input_shape[axis];
        }

        if constexpr (reduce_type == ReduceType::Sum) {
            output_tensor =
                reduce_sum(input_tensor, reduce_op_dim, scalar, memory_config, std::nullopt, compute_kernel_config);
        } else if constexpr (reduce_type == ReduceType::Mean) {
            output_tensor = reduce_sum(
                input_tensor, reduce_op_dim, 1.0 / reduced_volume, memory_config, std::nullopt, compute_kernel_config);
        } else if constexpr (reduce_type == ReduceType::Max) {
            output_tensor = reduce_with_padding(
                input_tensor,
                -std::numeric_limits<float>::infinity(),
                tt::tt_metal::ReduceOpMath::MAX,
                reduce_op_dim,
                scalar,
                memory_config,
                compute_kernel_config);
        } else if constexpr (reduce_type == ReduceType::Min) {
            output_tensor = reduce_with_padding(
                input_tensor,
                std::numeric_limits<float>::infinity(),
                tt::tt_metal::ReduceOpMath::MIN,
                reduce_op_dim,
                scalar,
                memory_config,
                compute_kernel_config);
        } else if constexpr (reduce_type == ReduceType::Var or reduce_type == ReduceType::Std) {
            auto mean_tensor = reduce_sum(
                input_tensor, reduce_op_dim, 1.0 / reduced_volume, memory_config, std::nullopt, compute_kernel_config);
            auto mean_square_tensor = reduce_sum(
                ttnn::pow(input_tensor, 2.0f, memory_config),
                reduce_op_dim,
                1.0 / reduced_volume,
                memory_config,
                std::nullopt,
                compute_kernel_config);
            output_tensor = ttnn::subtract(
                mean_square_tensor, ttnn::pow(mean_tensor, 2.0f, memory_config), std::nullopt, memory_config);
            if constexpr (reduce_type == ReduceType::Std) {
                output_tensor = ttnn::sqrt(output_tensor, memory_config);
            }
        } else {
            TT_THROW("Unsupported reduction operation");
        }
    }

    if (reshape) {
        if (output_shape.size() == 1) {  // Work around reshape not working for tile layouts when shape is of rank 1.
            output_tensor =
                ttnn::to_layout(output_tensor, Layout::ROW_MAJOR, std::nullopt, std::nullopt, output_tensor.device());
        }
        // output_tensor = ttnn::reshape(output_tensor, ttnn::Shape{output_shape, padded_output_shape});
        output_tensor = ttnn::reshape(output_tensor, ttnn::Shape{output_shape});
        if (output_shape.size() == 1 &&
            keepdim) {  // If do not keep dim, leave in row major. Tile layout needs rank of at least 2.
            output_tensor =
                ttnn::to_layout(output_tensor, Layout::TILE, std::nullopt, std::nullopt, output_tensor.device());
        }
    }
    return output_tensor;
}

template <ReduceType reduce_type>
Tensor Reduce<reduce_type>::invoke(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, ttnn::SmallVector<int>>>& dim_arg,
    const bool keepdim,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar) {
    return reduce_impl<reduce_type>(
        input_tensor_arg, dim_arg, keepdim, memory_config_arg, compute_kernel_config, scalar, true);
}

template class Reduce<ReduceType::Sum>;
template class Reduce<ReduceType::Mean>;
template class Reduce<ReduceType::Max>;
template class Reduce<ReduceType::Min>;
template class Reduce<ReduceType::Std>;
template class Reduce<ReduceType::Var>;
}  // namespace operations::reduction
}  // namespace ttnn
