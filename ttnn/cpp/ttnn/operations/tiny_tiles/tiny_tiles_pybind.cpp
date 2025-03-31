// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tiny_tiles_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sdpa_decode/sdpa_decode_pybind.hpp"

namespace ttnn::operations::tiny_tiles {

namespace py = pybind11;

void py_module(py::module& module) { py_bind_sdpa_decode(module); }

}  // namespace ttnn::operations::tiny_tiles
