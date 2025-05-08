// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sliced_reduce_scatter_async_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/sliced_reduce_scatter/sliced_reduce_scatter_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace detail {

template <typename ccl_operation_t>
void bind_sliced_reduce_scatter_async(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    namespace py = pybind11;

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               ttnn::Tensor& persistent_intermediate_buffer,
               ttnn::Tensor& persistent_output_buffer,
               const int32_t scatter_dim,
               const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const ttnn::ccl::Topology topology,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    persistent_intermediate_buffer,
                    persistent_output_buffer,
                    scatter_dim,
                    multi_device_global_semaphore,
                    num_links,
                    memory_config,
                    topology,
                    subdevice_id);
            },
            py::arg("input_tensor"),
            py::arg("persistent_intermediate_buffer"),
            py::arg("persistent_output_buffer"),
            py::arg("scatter_dim"),
            py::arg("multi_device_global_semaphore"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Ring,
            py::arg("subdevice_id") = std::nullopt});
}

}  // namespace detail

void py_bind_sliced_reduce_scatter_async(pybind11::module& module) {
    detail::bind_sliced_reduce_scatter_async(
        module,
        ttnn::experimental::sliced_reduce_scatter_async,
        R"doc(
        TODO
        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
