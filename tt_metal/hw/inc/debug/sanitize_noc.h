// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// debug/sanitize_noc.h
//
// This file implements a method sanitize noc addresses.
// Malformed addresses (out of range offsets, bad XY, etc) are stored in L1
// where the watcher thread can log the result.  The device then soft-hangs in
// a spin loop.
//
// All functionaly gated behind defined WATCHER_ENABLED
//
#pragma once

// NOC logging enabled independently of watcher, need to include it here because it hooks into DEBUG_SANITIZE_NOC_*
#include "noc_logging.h"

static tt_l1_ptr uint32_t* scratch_ptr = (uint32_t*)(0x21d0);
static tt_l1_ptr uint32_t* scratch_ptr_start = (uint32_t*)(0x21d0);

static uint32_t txn_record_ptr = 0;
#include "watcher_common.h"

#include "dev_msgs.h"
#include "noc_overlay_parameters.h"
#include "noc_parameters.h"
#include "noc_nonblocking_api.h"

// A couple defines for specifying read/write and multi/unicast
#define DEBUG_SANITIZE_NOC_READ true
#define DEBUG_SANITIZE_NOC_WRITE false
typedef bool debug_sanitize_noc_dir_t;
#define DEBUG_SANITIZE_NOC_MULTICAST true
#define DEBUG_SANITIZE_NOC_UNICAST false
typedef bool debug_sanitize_noc_cast_t;
#define DEBUG_SANITIZE_NOC_TARGET true
#define DEBUG_SANITIZE_NOC_LOCAL false
typedef bool debug_sanitize_noc_which_core_t;

// Helper function to get the core type from noc coords.

// Return value is the alignment mask for the type of core the noc address points
// to. Need to do this because L1 alignment needs to match the noc address alignment requirements,
// even if it's different than the inherent L1 alignment requirements.
// Direction is specified because reads and writes may have different L1 requirements (see noc_parameters.h).
FORCE_INLINE void debug_sanitize_noc_addr(
    uint8_t noc_id,
    uint64_t noc_addr,
    uint32_t l1_addr,
    uint32_t noc_len,
    debug_sanitize_noc_cast_t multicast,
    debug_sanitize_noc_dir_t dir) {
    txn_record_ptr += 1;
    *scratch_ptr = (uint32_t)noc_id;
    *(scratch_ptr + 1) = (uint32_t)(noc_addr >> 32);
    *(scratch_ptr + 2) = (uint32_t)(noc_addr);
    *(scratch_ptr + 3) = 0xffffffff;
    *(scratch_ptr + 4) = noc_len;
    *(scratch_ptr + 5) = (uint32_t)multicast;
    *(scratch_ptr + 6) = (uint32_t)dir;
    scratch_ptr += 7;
    if (txn_record_ptr == 6) {
        txn_record_ptr = 0;
        scratch_ptr = scratch_ptr_start;
    }
}

FORCE_INLINE void debug_sanitize_noc_and_worker_addr(
    uint8_t noc_id,
    uint64_t noc_addr,
    uint32_t worker_addr,
    uint32_t len,
    debug_sanitize_noc_cast_t multicast,
    debug_sanitize_noc_dir_t dir) {
    txn_record_ptr += 1;
    *scratch_ptr = (uint32_t)noc_id;
    *(scratch_ptr + 1) = (uint32_t)(noc_addr >> 32);
    *(scratch_ptr + 2) = (uint32_t)(noc_addr);
    *(scratch_ptr + 3) = worker_addr;
    *(scratch_ptr + 4) = len;
    *(scratch_ptr + 5) = (uint32_t)multicast;
    *(scratch_ptr + 6) = (uint32_t)dir;
    scratch_ptr += 7;
    if (txn_record_ptr == 6) {
        txn_record_ptr = 0;
        scratch_ptr = scratch_ptr_start;
    }
}

// TODO: Clean these up with #7453
#define DEBUG_SANITIZE_NOC_ADDR(noc_id, a, l)
#define DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc_id, noc_a, worker_a, l)

#define DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc_id, noc_a, worker_a, l)

#define DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_FROM_STATE(noc_id)
#define DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc_id, noc_a_lower, worker_a)
#define DEBUG_SANITIZE_NOC_ADDR_FROM_STATE(noc_id, cmd_buf)
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc_id, noc_a, worker_a, l)
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_AND_SIZE_STATE(noc_id, noc_a_lower, worker_a)
#define DEBUG_SANITIZE_NOC_READ_TRANSACTION_WITH_ADDR_STATE(noc_id, noc_a_lower, worker_a, l)
#define DEBUG_INSERT_DELAY(transaction_type)
