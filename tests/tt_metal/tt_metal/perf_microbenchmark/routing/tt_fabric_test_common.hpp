// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

/* Common file for host and device */

// keep any semaphore size as 16 bytes for noc alignment
static constexpr std::uint32_t SEMAPHORE_SIZE_BYTES = 16;

static constexpr std::uint32_t TEST_RESULTS_SIZE_BYTES = 128;
static constexpr std::uint32_t TEST_RESULTS_ADDRESS_OFFSET = 0;
static constexpr std::uint32_t HOST_TO_CONTROLLER_SEM_OFFSET = TEST_RESULTS_ADDRESS_OFFSET + TEST_RESULTS_SIZE_BYTES;
static constexpr std::uint32_t CONTROLLER_TO_CONTROLLER_SEM_OFFSET =
    HOST_TO_CONTROLLER_SEM_OFFSET + SEMAPHORE_SIZE_BYTES;
static constexpr std::uint32_t CONTROLLER_TO_WORKERS_SEM_OFFSET =
    CONTROLLER_TO_CONTROLLER_SEM_OFFSET + SEMAPHORE_SIZE_BYTES;
static constexpr std::uint32_t SENDERS_TO_CONTROLLER_SEM_OFFSET =
    CONTROLLER_TO_WORKERS_SEM_OFFSET + SEMAPHORE_SIZE_BYTES;
static constexpr std::uint32_t RECEIVERS_TO_CONTROLLER_SEM_OFFSET =
    SENDERS_TO_CONTROLLER_SEM_OFFSET + SEMAPHORE_SIZE_BYTES;
static constexpr std::uint32_t WORKER_USABLE_BASE_ADDRESS_OFFSET =
    RECEIVERS_TO_CONTROLLER_SEM_OFFSET + SEMAPHORE_SIZE_BYTES;
static constexpr std::uint32_t BASE_TARGET_ADDRESS_OFFSET = WORKER_USABLE_BASE_ADDRESS_OFFSET;

static constexpr std::uint8_t NUM_DIRECTIONS = 4;
static constexpr std::uint8_t MAX_NUM_SENDERS_PER_RECEIVER = 4;
static constexpr std::uint32_t INVALID_SENDER_ID = std::numeric_limits<uint32_t>::max();
static constexpr std::uint32_t L1_BUFFER_SIZE_PER_SENDER_BYTES = 28 * 1024;
