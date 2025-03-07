// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/common/thread_pool.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal::distributed {
namespace {

// Stress test for thread pool used by TT-Mesh
TEST(ThreadPoolTest, Stress) {
    // Enqueue enough tasks to saturate the thread pool.
    uint64_t NUM_ITERS = 200;
    auto thread_pool = create_boost_thread_pool(8);
    // Increment this once for each task in the thread pool.
    // Use this to verify that tasks actually executed.
    std::atomic<uint64_t> counter = 0;
    auto incrementer_fn = [&counter]() {
        counter++;
        // Sleep every 10 iterations to slow down the workers - allows
        // the thread pool to get saturated
        if (counter.load() % 10 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        }
    };

    for (std::size_t iter = 0; iter < NUM_ITERS; iter++) {
        thread_pool->enqueue([&incrementer_fn]() mutable { incrementer_fn(); });
    }
    thread_pool->wait();
    for (std::size_t iter = 0; iter < NUM_ITERS; iter++) {
        thread_pool->enqueue([&incrementer_fn]() mutable { incrementer_fn(); });
    }

    thread_pool->wait();
    EXPECT_EQ(counter.load(), 2 * NUM_ITERS);
}

}  // namespace
}  // namespace tt::tt_metal::distributed
