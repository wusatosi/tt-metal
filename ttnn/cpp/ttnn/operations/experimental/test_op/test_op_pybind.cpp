// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/pybind11/decorators.hpp"

#include "ttnn/operations/experimental/test_op/test_op.hpp"
#include "ttnn/operations/experimental/test_op/test_op_pybind.hpp"

#include <fmt/format.h>

namespace ttnn::operations::experimental::test_op::detail {
namespace py = pybind11;

void bind_experimental_test_op_operation(py::module& module) {
    auto doc = fmt::format(R"doc(test_op)doc");

    using OperationType = decltype(ttnn::experimental::test_op);
    bind_registered_operation(
        module,
        ttnn::experimental::test_op,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& inp0,
               const Tensor& inp1,
               const string& metadata,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor>& output,
               QueueId queue_id) -> ttnn::Tensor {
                return self(queue_id, inp0, inp1, metadata, memory_config, output);
            },
            py::arg("inp0"),
            py::arg("inp1"),
            py::kw_only(),
            py::arg("metadata") = "none",
            py::arg("memory_config") = std::nullopt,
            py::arg("output") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}
}  // namespace ttnn::operations::experimental::test_op::detail
