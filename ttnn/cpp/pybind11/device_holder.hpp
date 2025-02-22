// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tt-metalium/device_impl.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <memory>
#include <type_traits>

namespace ttnn {

template <typename T>
class DeviceHolder {
public:
    using element_type = T;

    DeviceHolder(T* ptr) {
        value = ptr;
        if constexpr (std::is_same_v<T, tt::tt_metal::distributed::MeshDevice>) {
            owner = ptr->shared_from_this();
        } else if constexpr (std::is_same_v<T, tt::tt_metal::IDevice>) {
            if (auto mesh = dynamic_cast<tt::tt_metal::distributed::MeshDevice*>(ptr)) {
                owner = mesh->shared_from_this();
            }
        }
    }

    DeviceHolder(std::shared_ptr<tt::tt_metal::distributed::MeshDevice> shared) {
        owner = std::move(shared);
        value = owner.get();
    }

    DeviceHolder() = default;
    DeviceHolder(const DeviceHolder<T>&) = default;
    DeviceHolder& operator=(const DeviceHolder<T>&) = default;
    DeviceHolder(DeviceHolder<T>&&) = default;
    DeviceHolder& operator=(DeviceHolder<T>&&) = default;

    T* get() const { return value; }

private:
    T* value = nullptr;
    std::shared_ptr<T> owner;
};

}  // namespace ttnn

PYBIND11_DECLARE_HOLDER_TYPE(T, ttnn::DeviceHolder<T>, true);
