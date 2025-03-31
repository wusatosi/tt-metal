// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sdpa_decode.hpp"
#include "cpp/pybind11/decorators.hpp"

namespace ttnn::operations::tiny_tiles {

void py_bind_sdpa_decode(py::module& module) {
    auto doc =
        R"doc(
        A simplified version of sdpa decode for testing LLK tiny tiles.
        )doc";

    using OperationType = decltype(ttnn::tiny_tiles::sdpa_decode);
    ttnn::bind_registered_operation(
        module,
        ttnn::tiny_tiles::sdpa_decode,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor_q,
               const ttnn::Tensor& input_tensor_k,
               const ttnn::Tensor& input_tensor_v,
               QueueId queue_id) { return self(queue_id, input_tensor_q, input_tensor_k, input_tensor_v); },
            py::arg("input_tensor_q").noconvert(),
            py::arg("input_tensor_k").noconvert(),
            py::arg("input_tensor_v").noconvert(),
            py::kw_only(),
            py::arg("queue_id") = DefaultQueueId,
        });
}
}  // namespace ttnn::operations::tiny_tiles
