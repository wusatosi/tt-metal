// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "muladd_pybind.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/decorators.hpp"
#include "tt-metalium/base_types.hpp"
#include "ttnn/operations/muladd/muladd.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::muladd {

namespace py = pybind11;

void py_module(py::module& module) {
    using OperationType = decltype(ttnn::muladd);
    const auto doc = R"doc(
        Performs (A+B)*C/D elementwise for input vectors A,B,C,D with same dimensions (currently only (32,32))
    )doc";

    ttnn::bind_registered_operation(
        module,
        ttnn::muladd,
        doc,
        ttnn::pybind_overload_t{
            [](const decltype(::ttnn::muladd)& self,
               const ttnn::Tensor& inputA,
               const ttnn::Tensor& inputB,
               const ttnn::Tensor& inputC,
               const ttnn::Tensor& inputD,
               const MemoryConfig& memory_config,
               const DataType& dtype,
               const MathFidelity& math_fidelity,
               QueueId queue_id) -> ttnn::Tensor {
                return self(queue_id, inputA, inputB, inputC, inputD, memory_config, dtype, math_fidelity);
            },
            py::arg("inputA"),
            py::arg("inputB"),
            py::arg("inputC"),
            py::arg("inputD"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("math_fidelity") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::muladd
