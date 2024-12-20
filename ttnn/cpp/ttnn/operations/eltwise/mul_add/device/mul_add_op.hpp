// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sys/types.h>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

#include "tt_metal/host_api.hpp"

namespace ttnn::operations::mul_add {

struct MulAddDeviceOperation {
    struct operation_attributes_t {
        bool attribute;
    };

    struct tensor_args_t {
        const Tensor& input_tensor_a;
        const Tensor& input_tensor_b;
        const Tensor& input_tensor_c;
    };

    using tensor_return_value_t = Tensor;
    using spec_return_value_t = TensorSpec;

    struct MulAddProgramFactoryMultiCore {
        struct shared_variables_t {
            KernelHandle reader_kernel_id;
            KernelHandle writer_kernel_id;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            Tensor& tensor_return_value);
    };

    struct MulAddProgramFactoryMultiCoreSharded {
        struct shared_variables_t {
            KernelHandle reader_kernel_id;
            KernelHandle writer_kernel_id;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            tt::tt_metal::Program program{};
            return {std::move(program), {.reader_kernel_id = 1, .writer_kernel_id = 2}};
        }

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {}
    };

    using program_factory_t = std::variant<MulAddProgramFactoryMultiCore, MulAddProgramFactoryMultiCoreSharded>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor_a, const Tensor& input_tensor_b, const Tensor& input_tensor_c);
};

}  // namespace ttnn::operations::mul_add

namespace ttnn::prim {

constexpr auto muladd =
    ttnn::register_operation<"ttnn::prim::muladd", ttnn::operations::mul_add::MulAddDeviceOperation>();
}  // namespace ttnn::prim
