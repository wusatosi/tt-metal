// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-pybind/decorators.hpp"

#include "ttnn/operations/experimental/sdxl_group_norm/group_norm.hpp"
#include "ttnn/operations/experimental/sdxl_group_norm/group_norm_pybind.hpp"

namespace ttnn::operations::experimental::sdxl_group_norm::detail {
namespace py = pybind11;
void bind_experimental_group_norm_operation(py::module& module) {
    auto doc =
        R"doc(plus_one(input_tensor: ttnn.Tensor) -> ttnn.Tensor

            Returns input tensor elements increased by 1.
            Input tensor must have UINT32 data type, ROW_MAJOR layout, and 1-D shape.
            This op only gives decent performance for small tensors (up to 100 elements).
            Specify the core to use in the sub_core_grids argument.

            Equivalent pytorch code:

            .. code-block:: python

                return torch.add(input_tensor, 1)

            Args:
                * :attr:`input_tensor`: Input Tensor for plusone.

        )doc";

    using OperationType = decltype(ttnn::experimental::sdxl_group_norm);
    bind_registered_operation(
        module,
        ttnn::experimental::sdxl_group_norm,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weights,
               const ttnn::Tensor& bias,
               const float eps,
               const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
               const std::optional<CoreRangeSet>& sub_core_grids) {
                return self(input_tensor, weights, bias, eps, compute_kernel_config, sub_core_grids);
            },
            py::arg("input_tensor").noconvert(),
            py::kw_only(),
            py::arg("weights").noconvert(),
            py::arg("bias").noconvert(),
            py::arg("eps") = 1e-05,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("sub_core_grids") = std::nullopt});
}

}  // namespace ttnn::operations::experimental::sdxl_group_norm::detail
