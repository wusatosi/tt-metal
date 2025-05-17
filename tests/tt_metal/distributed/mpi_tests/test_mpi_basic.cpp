// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <fmt/format.h>
#include <tt_stl/tt_stl/reflection.hpp>

#include <tt-metalium/distributed_context.hpp>  
#include <vector>

int main(int argc, char **argv) {
    tt::tt_metal::distributed::multihost::DistributedContext::create(argc, argv);
    auto context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    using Rank = tt::tt_metal::distributed::multihost::Rank;
    using Tag = tt::tt_metal::distributed::multihost::Tag;
    fmt::print("Distributed context created\n");
    fmt::print("Rank: {}\n", context->rank());
    fmt::print("World size: {}\n", context->size());
    if (*context->rank() == 0) {
        std::vector<std::byte> bytes(10);
        for (int i = 0; i < 10; ++i) {
            bytes[i] = static_cast<std::byte>(i);
        }
        fmt::print("Sending bytes to rank 1\n");
        tt::stl::Span<std::byte> view(bytes.data(), bytes.size());
        context->send(view, Rank{1}, Tag{0});
    } else {
        fmt::print("Hello from rank {}\n", context->rank());
        std::vector<std::byte> bytes(10);
        fmt::print("Receiving bytes from rank 0\n");
        tt::stl::Span<std::byte> view(bytes.data(), bytes.size());
        context->recv(view, Rank{0}, Tag{0});
        fmt::print("Bytes: ");
        for (size_t i = 0; i < bytes.size(); ++i) {
            fmt::print("{}{}", static_cast<int>(bytes[i]), (i < bytes.size() - 1) ? ", " : "\n");
        }
        
    }
}