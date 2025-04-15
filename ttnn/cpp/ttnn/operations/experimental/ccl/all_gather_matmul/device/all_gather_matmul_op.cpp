// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul/device/all_gather_matmul_op.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

/* All Gather Matmul fusion includes */
#include "cpp/ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "cpp/ttnn/operations/matmul/device/matmul_op.hpp"
#include "cpp/ttnn/operations/matmul/matmul.hpp"

namespace ttnn {
namespace experimental {

void AllGatherMatmul::validate(
    const AllGatherMatmul::operation_attributes_t& attr, const AllGatherMatmul::tensor_args_t& tensor_args) {
    auto& input_tensor = tensor_args.input_tensor;
    auto& all_gather_output_tensor = tensor_args.all_gather_output;
    auto& weight_tensor = tensor_args.weight_tensor;

    // All Gather validate
    attr.all_gather_struct.validate({input_tensor});

    // Matmul validate.
    operations::matmul::Matmul::validate(
        attr.matmul_struct,
        {
            .input_tensor_a = all_gather_output_tensor,
            .input_tensor_b = weight_tensor,
        });

    // All Gather Matmul validate
    TT_FATAL(attr.all_gather_struct.dim == 3, "AllGatherMatmul requires dim=3 for the AllGather operaitons.");
    TT_FATAL(
        input_tensor.get_padded_shape()[0] == 1 && input_tensor.get_padded_shape()[1] == 1,
        "AllGatherMatmul requires input tensor to have batch size of 1.");
    std::visit(
        [&](const auto& config) {
            using ProgramConfigType = std::decay_t<decltype(config)>;
            if (not(std::is_same_v<
                        ProgramConfigType,
                        operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig> ||
                    std::
                        is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>)) {
                TT_THROW("Unsupported MatmulProgramConfig type for AllGatherMatmul. Needs to be 1D or 2D Multicast.");
            }
        },
        attr.matmul_struct.program_config.value());

    const auto& all_gather_output_tensor_shard_spec = all_gather_output_tensor.shard_spec();
    if (all_gather_output_tensor_shard_spec.has_value()) {
        auto const& shard_grid = all_gather_output_tensor_shard_spec->grid.bounding_box();
        auto const& shard_grid_start = shard_grid.start_coord;
        auto const& shard_grid_end = shard_grid.end_coord;
        const uint32_t num_all_gather_output_shards = shard_builder::get_sharding_core_count(all_gather_output_tensor);
        TT_FATAL(
            attr.all_gather_struct.ring_size == num_all_gather_output_shards,
            "AllGatherMatmul requires number of tensor slices to equal the number of output shards of the all_gather.");
    }
}

std::vector<ttnn::TensorSpec> AllGatherMatmul::compute_output_specs(
    const AllGatherMatmul::operation_attributes_t& attr, const AllGatherMatmul::tensor_args_t& tensor_args) {
    // All Gather shape
    ttnn::TensorSpec all_gather_output_shape =
        attr.all_gather_struct.compute_output_specs({tensor_args.input_tensor})[0];
    ttnn::TensorSpec datacopy_output_shape = all_gather_output_shape;

    // Matmul shape
    ttnn::TensorSpec matmul_output_specs = operations::matmul::Matmul::compute_output_specs(
        attr.matmul_struct,
        {.input_tensor_a = tensor_args.all_gather_output, .input_tensor_b = tensor_args.weight_tensor});

    return {all_gather_output_shape, matmul_output_specs, datacopy_output_shape};
}

std::vector<Tensor> AllGatherMatmul::create_output_tensors(
    const AllGatherMatmul::operation_attributes_t& attr, const AllGatherMatmul::tensor_args_t& tensor_args) {
    // All Gather output tensor
    auto& all_gather_output_tensor = tensor_args.all_gather_output;
    auto& datacopy_output_tensor = tensor_args.datacopy_output;

    // Matmul output tensor
    ttnn::Tensor matmul_output_tensor = operations::matmul::Matmul::create_output_tensors(
        attr.matmul_struct,
        {.input_tensor_a = tensor_args.all_gather_output, .input_tensor_b = tensor_args.weight_tensor});

    return {all_gather_output_tensor, matmul_output_tensor, datacopy_output_tensor};
}

tt::tt_metal::ProgramDescriptor AllGatherMatmul::create_program(
    const AllGatherMatmul::operation_attributes_t& attr,
    const AllGatherMatmul::tensor_args_t& tensor_args,
    std::vector<Tensor>& output_tensors) {
    // Return the AllGatherMatmul program with callbacks
    return all_gather_matmul_multi_core_with_workers(
        tensor_args.input_tensor,   // input_tensor
        output_tensors[0],          // all_gather_output_tensor
        output_tensors[2],          // datacopy_output_tensor
        tensor_args.weight_tensor,  // weight_tensor
        output_tensors[1],          // matmul_output_tensor

        /* All Gather Params */
        attr.all_gather_struct.dim,
        attr.all_gather_struct.num_links,
        attr.all_gather_struct.ring_size,
        attr.all_gather_struct.ring_index,
        attr.all_gather_struct.user_defined_num_workers,
        attr.all_gather_struct.user_defined_num_buffers_per_channel,
        attr.all_gather_struct.receiver_device_id,
        attr.all_gather_struct.sender_device_id,
        attr.all_gather_struct.topology,
        attr.all_gather_core_grid_offset,

        /* Matmul Params */
        {},  // Bias
        attr.matmul_struct.bcast_batch.value(),
        attr.matmul_struct.compute_kernel_config.value(),
        attr.matmul_struct.program_config.value(),
        attr.matmul_struct.untilize_out);
}

std::tuple<AllGatherMatmul::operation_attributes_t, AllGatherMatmul::tensor_args_t> AllGatherMatmul::invoke(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const uint32_t dim,
    const CoreCoord all_gather_core_grid_offset,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config_ag,
    std::optional<size_t> user_defined_num_workers,
    std::optional<size_t> user_defined_num_buffers_per_channel,
    const std::optional<MemoryConfig>& memory_config_mm,
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const DataType> dtype,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const std::string>& activation,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const ttnn::CoreGrid> core_grid) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "AllGatherMatmul is only supported for Fast Dispatch");

    auto devices = input_tensor.get_workers();

    /* AllGather setup */
    ttnn::AllGather all_gather_struct = ttnn::ccl::all_gather_detail::create_all_gather_struct(
        input_tensor,
        dim,
        num_links,
        memory_config_ag,
        user_defined_num_workers,
        user_defined_num_buffers_per_channel,
        devices,
        ttnn::ccl::Topology::Ring);

    // Create the all gather output tensor used as input (activation) to the matmul
    ttnn::Tensor all_gather_out_tensor = all_gather_struct.create_output_tensors({input_tensor})[0];
    ttnn::Tensor datacopy_out_tensor = all_gather_struct.create_output_tensors({input_tensor})[0];

    /* Matmul setup */
    bool user_run_batched = ttnn::operations::matmul::detail::is_input_batched(weight_tensor.get_logical_shape());
    std::optional<CoreCoord> user_core_coord;
    if (core_grid.has_value()) {
        user_core_coord = CoreCoord(core_grid->x, core_grid->y);
    }

    auto matmul_struct = operations::matmul::create_matmul_struct(
        all_gather_out_tensor,
        weight_tensor,
        /*parameters=*/
        operations::matmul::MatmulArgs{
            program_config,
            /*bcast_batch=*/std::nullopt,
            memory_config_mm.value_or(input_tensor.memory_config()),
            dtype.value_or(input_tensor.get_dtype()),
            compute_kernel_config,
            /*untilize_out=*/false,
            user_core_coord,
            ttnn::operations::matmul::get_fused_activation(activation),
            user_run_batched,
            transpose_a,
            transpose_b,
            /*output_tile=*/std::nullopt,
            /*global_cb=*/std::nullopt});

    AllGatherMatmul::operation_attributes_t all_gather_matmul_struct = {/* All Gather Params */
                                                                        all_gather_struct,
                                                                        /* Matmul params */
                                                                        matmul_struct,
                                                                        /* Fusion params */
                                                                        all_gather_core_grid_offset};

    return {all_gather_matmul_struct, {input_tensor, all_gather_out_tensor, weight_tensor, datacopy_out_tensor}};
}

}  // namespace experimental

}  // namespace ttnn
