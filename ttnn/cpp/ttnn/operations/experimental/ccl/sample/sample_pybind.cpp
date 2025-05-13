// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sample_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/sample/sample.hpp"
#include "ttnn/operations/experimental/ccl/sample/sample_pybind.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::ccl {

template <typename ccl_operation_t>
void py_bind_sample_async(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    namespace py = pybind11;

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const std::vector<ttnn::GlobalSemaphore>& semaphores) -> ttnn::Tensor {
                return self(input_tensor, semaphores);
            },
        });
}

void py_bind_sample_async(pybind11::module& module) { py_bind_sample_async(module, ttnn::experimental::sample, ""); }

}  // namespace ttnn::operations::experimental::ccl
