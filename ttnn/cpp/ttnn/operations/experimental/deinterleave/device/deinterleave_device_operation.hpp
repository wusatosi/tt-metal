// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deinterleave {

struct DeinterleaveOperation {
    struct operation_attributes_t {
        const uint32_t input_height;
        const uint32_t input_width;
        const std::array<uint32_t, 2> stride_hw;
        const bool to_batch;
        const uint32_t barrier_threshold;
        const DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& input;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactoryToBatch {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle read_kernel_id;
            tt::tt_metal::KernelHandle write_kernel_id;
            tt::tt_metal::CoreRangeSet worker_grid;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    struct ProgramFactoryLocal {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle read_kernel_id;
            tt::tt_metal::KernelHandle write_kernel_id;
            tt::tt_metal::CoreRangeSet worker_grid;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };
    using program_factory_t = std::variant<ProgramFactoryToBatch, ProgramFactoryLocal>;

    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const uint32_t input_height,
        const uint32_t input_width,
        const std::array<uint32_t, 2> stride_hw,
        const bool to_batch,
        const uint32_t barrier_threshold,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};

}  // namespace ttnn::operations::experimental::deinterleave

namespace ttnn::prim {
constexpr auto deinterleave = ttnn::register_operation<
    "ttnn::prim::deinterleave",
    ttnn::operations::experimental::deinterleave::DeinterleaveOperation>();
}  // namespace ttnn::prim
