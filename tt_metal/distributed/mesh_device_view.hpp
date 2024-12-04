// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <span>
#include <optional>
#include <functional>

#include "tt_metal/impl/device/device.hpp"

namespace tt::tt_metal::distributed {

// Forward declaration of MeshDevice
class MeshDevice;

class MeshShape {
public:
    static constexpr size_t MAX_RANK = 10;

    MeshShape();
    explicit MeshShape(std::initializer_list<uint32_t> ilist);
    MeshShape(const std::span<const uint32_t>& span);

    // Support std::pair initialization for backwards compatibility
    template <typename T>
    MeshShape(const std::pair<T, T>& shape) :
        MeshShape({static_cast<uint32_t>(shape.first), static_cast<uint32_t>(shape.second)}) {}

    MeshShape& operator=(std::initializer_list<uint32_t> ilist);
    MeshShape& operator=(const std::span<const uint32_t>& span);
    bool operator==(const MeshShape &other) const;
    uint32_t operator[](int32_t index) const;
    uint32_t& operator[](int32_t index);

    size_t rank() const;
    uint64_t volume() const;

    auto cbegin() const;
    auto cend() const;

    static constexpr auto attribute_names = std::forward_as_tuple("dims", "rank");
    constexpr auto attribute_values() const { return std::forward_as_tuple(dims, rank_size); }

private:
    std::array<uint32_t, MAX_RANK> dims;
    size_t rank_size;
};

struct Coordinate {
    size_t row;
    size_t col;
    auto operator<=>(const Coordinate&) const = default;

    // Add support for structured bindings
    template <size_t I>
    decltype(auto) get() const {
        if constexpr (I == 0) return row;
        else if constexpr (I == 1) return col;
        else static_assert(I < 2, "Index out of bounds for Coordinate");
    }

    friend std::ostream& operator<<(std::ostream& os, const Coordinate& coord) {
        return os << "Coord(" << coord.row << ", " << coord.col << ")";
    }
};

/**
 * @brief The MeshDeviceView class provides a view of a specific sub-region within the MeshDevice.
 *
 * Once a MeshDevice is initialized, MeshDeviceView allows the creation of multiple "views" on the
 * MeshDevice, enabling more granular control over a cluster of initialized devices. This approach
 * differs from simply creating a new MeshDevice on a subset of devices.
 *
 * MeshDeviceView serves two primary purposes:
 *
 * 1. It facilitates the creation of abstractions that define parallelization strategies, such as
 *    tensor-parallel or pipeline-parallel, by assigning views of the MeshDevice.
 *
 * 2. It acts as a query interface for the MeshDevice, allowing the retrieval of devices based on
 *    specific sub-regions. This is particularly useful for collective communication operations
 *    (CCL-ops), such as line all-gather, which require column or row views of the device mesh.
 */

enum class MeshType {
    RowMajor,
    Ring,
    Line
};

class MeshDeviceView {
public:
    using device_pointer = Device*;
    using const_device_pointer = const Device*;
    using DeviceView = std::vector<device_pointer>;
    using DeviceViews = std::vector<std::vector<device_pointer>>;
    using CoordinateMapper = std::function<std::optional<Coordinate>(int device_id)>;

    MeshDeviceView(const MeshDevice& mesh);
    MeshDeviceView(const MeshDevice& mesh, Coordinate top_left, Coordinate bottom_right);
    MeshDeviceView(std::vector<device_pointer> devices, CoordinateMapper mapper);

    [[nodiscard]] device_pointer get_device(size_t row, size_t col);
    [[nodiscard]] const_device_pointer get_device(size_t row, size_t col) const;

    // Get devices spanning the rectangular region defined by the top-left and bottom-right coordinates
    // devices are returned in row-major order with start/end coordinates inclusive
    [[nodiscard]] DeviceView get_devices(const Coordinate& start, const Coordinate& end);
    [[nodiscard]] DeviceView get_devices(const MeshShape& shape);
    [[nodiscard]] DeviceView get_devices(MeshType type = MeshType::RowMajor);

    [[nodiscard]] DeviceView get_devices_on_row(size_t row) const;
    [[nodiscard]] DeviceView get_devices_on_column(size_t col) const;

    [[nodiscard]] DeviceViews get_row_views() const;
    [[nodiscard]] DeviceViews get_column_views() const;

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] size_t size() const noexcept;
    [[nodiscard]] MeshShape shape() const noexcept;
    [[nodiscard]] bool contains(const Coordinate& coord) const noexcept;
    [[nodiscard]] const_device_pointer at(const Coordinate& coord) const noexcept;

    bool operator==(const MeshDeviceView& other) const;

    auto begin() const { return devices_.begin(); }
    auto end() const { return devices_.end(); }

    [[nodiscard]] size_t num_rows() const { return bottom_right_.row - top_left_.row + 1; }
    [[nodiscard]] size_t num_cols() const { return bottom_right_.col - top_left_.col + 1; }
    [[nodiscard]] size_t num_devices() const { return devices_.size(); }

    [[nodiscard]] bool contains_device(chip_id_t device_id) const;
    [[nodiscard]] Coordinate find_device(chip_id_t device_id) const;
    [[nodiscard]] chip_id_t find_device_id(const Coordinate& coord) const;

    // Given a starting coordinate, get the coordinates of a line of devices where device[i-1] is connected to device[i]
    // The current support only provides left-to-right and right-to-left snaking of the line.
    [[nodiscard]] static std::vector<Coordinate> get_line_coordinates(size_t length, const Coordinate& offset, size_t num_rows, size_t num_cols);
    [[nodiscard]] std::vector<Coordinate> get_ring_coordinates(const MeshShape& ring_shape, const Coordinate& offset, size_t num_rows, size_t num_cols);
    [[nodiscard]] std::vector<device_pointer> get_ring_devices();
    [[nodiscard]] std::vector<device_pointer> get_line_devices();

private:
    std::vector<device_pointer> devices_;
    std::unordered_map<chip_id_t, Coordinate> device_coordinates_;
    Coordinate top_left_;
    Coordinate bottom_right_;

    void initialize_from_devices(const std::vector<device_pointer>& devices, CoordinateMapper mapper);
    void validate_coordinates() const;
};

} // namespace tt::tt_metal::distributed

namespace std {
    // Specializations to enable structured bindings
    template<> struct tuple_size<tt::tt_metal::distributed::Coordinate> : std::integral_constant<size_t, 2> {};
    template<size_t I> struct tuple_element<I, tt::tt_metal::distributed::Coordinate> {
        using type = size_t;
    };

    // Specialization to enable hashing of Coordinate
    template <>
    struct hash<tt::tt_metal::distributed::Coordinate> {
        size_t operator()(const tt::tt_metal::distributed::Coordinate& coord) const noexcept {
            size_t seed = 0;
            tt::utils::hash_combine(seed, coord.row);
            tt::utils::hash_combine(seed, coord.col);
            return seed;
        }
    };
} // namespace std
