// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>
#include <tuple>

#include "tt-metalium/core_coord.hpp"
#include "ttnn/types.hpp"

using DataChunk = std::tuple<CoreCoord, uint32_t, uint32_t>;
using CoreMapping = std::pair<CoreCoord, std::vector<DataChunk>>;

std::vector<CoreMapping> calculate_core_mapping(
    const ttnn::CoreRangeSet& cores,
    int block_size,
    uint32_t input_sticks_per_core,
    uint32_t output_blocks_per_core,
    uint32_t num_cores_with_extra_block);

std::vector<uint32_t> calculate_shard_sizes(
    const ttnn::CoreRangeSet& cores,
    int block_size,
    uint32_t input_sticks_per_core,
    uint32_t output_blocks_per_core,
    uint32_t num_cores_with_extra_block);
