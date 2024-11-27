// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _NOC_NON_BLOCKING_API_COMMON_H_
#define _NOC_NON_BLOCKING_API_COMMON_H_

#include <stdint.h>

#include <cstdint>

#include "noc_overlay_parameters.h"

#define   STREAM_RD_RESP_RECEIVED   STREAM_SCRATCH_0_REG_INDEX
#define   STREAM_NONPOSTED_WR_REQ_SENT   STREAM_SCRATCH_1_REG_INDEX
#define   STREAM_NONPOSTED_WR_ACK_RECEIVED   STREAM_SCRATCH_2_REG_INDEX
#define   STREAM_NONPOSTED_ATOMIC_RESP_RECEIVED   STREAM_SCRATCH_3_REG_INDEX
#define   STREAM_POSTED_WR_REQ_SENT   STREAM_SCRATCH_4_REG_INDEX

const uint32_t OPERAND_BRISC_NOC0_STREAM = 0;
const uint32_t OPERAND_BRISC_NOC1_STREAM = 1;
const uint32_t OPERAND_NCRISC_NOC0_STREAM = 2;
const uint32_t OPERAND_NCRISC_NOC1_STREAM = 3;

template<uint32_t risc_type>
inline __attribute__((always_inline)) uint32_t get_stream_index(uint32_t noc) {
  if constexpr (risc_type == static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0)) {
    return noc == 0 ? OPERAND_BRISC_NOC0_STREAM : OPERAND_BRISC_NOC1_STREAM;
  } else {
    return noc == 0 ? OPERAND_NCRISC_NOC0_STREAM : OPERAND_NCRISC_NOC1_STREAM;
  }
}

// noc_reads_num_issued
template<uint32_t risc_type>
inline __attribute__((always_inline)) uint32_t get_noc_reads_num_issued(uint32_t noc) {
  return NOC_STREAM_READ_REG(get_stream_index<risc_type>(noc), STREAM_RD_RESP_RECEIVED);
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void inc_noc_reads_num_issued(uint32_t noc, uint32_t inc = 1) {
  uint32_t stream_id = get_stream_index<risc_type>(noc);
  uint32_t val = NOC_STREAM_READ_REG(stream_id, STREAM_RD_RESP_RECEIVED) + inc;
  NOC_STREAM_WRITE_REG(stream_id, STREAM_RD_RESP_RECEIVED, val);
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void set_noc_reads_num_issued(uint32_t noc, uint32_t val) {
  NOC_STREAM_WRITE_REG(get_stream_index<risc_type>(noc), STREAM_RD_RESP_RECEIVED, val);
}

// noc_nonposted_writes_num_issued
template<uint32_t risc_type>
inline __attribute__((always_inline)) volatile uint32_t get_noc_nonposted_writes_num_issued(uint32_t noc) {
  return NOC_STREAM_READ_REG(get_stream_index<risc_type>(noc), STREAM_NONPOSTED_WR_REQ_SENT);
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void inc_noc_nonposted_writes_num_issued(uint32_t noc, uint32_t inc = 1) {
  uint32_t stream_id = get_stream_index<risc_type>(noc);
  uint32_t val = NOC_STREAM_READ_REG(stream_id, STREAM_NONPOSTED_WR_REQ_SENT) + inc;
  NOC_STREAM_WRITE_REG(stream_id, STREAM_NONPOSTED_WR_REQ_SENT, val);
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void set_noc_nonposted_writes_num_issued(int32_t noc, uint32_t val) {
  NOC_STREAM_WRITE_REG(get_stream_index<risc_type>(noc), STREAM_NONPOSTED_WR_REQ_SENT, val);
}

// noc_nonposted_writes_acked
template<uint32_t risc_type>
inline __attribute__((always_inline)) uint32_t get_noc_nonposted_writes_acked(uint32_t noc) {
  return NOC_STREAM_READ_REG(get_stream_index<risc_type>(noc), STREAM_NONPOSTED_WR_ACK_RECEIVED);
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void inc_noc_nonposted_writes_acked(uint32_t noc, uint32_t inc = 1) {
  uint32_t stream_id = get_stream_index<risc_type>(noc);
  uint32_t val = NOC_STREAM_READ_REG(stream_id, STREAM_NONPOSTED_WR_ACK_RECEIVED) + inc;
  NOC_STREAM_WRITE_REG(stream_id, STREAM_NONPOSTED_WR_ACK_RECEIVED, val);
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void set_noc_nonposted_writes_acked(uint32_t noc, uint32_t val) {
  NOC_STREAM_WRITE_REG(get_stream_index<risc_type>(noc), STREAM_NONPOSTED_WR_ACK_RECEIVED, val);
}

// noc_nonposted_atomics_acked
template<uint32_t risc_type>
inline __attribute__((always_inline)) uint32_t get_noc_nonposted_atomics_acked(uint32_t noc) {
  return NOC_STREAM_READ_REG(get_stream_index<risc_type>(noc), STREAM_NONPOSTED_ATOMIC_RESP_RECEIVED);
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void inc_noc_nonposted_atomics_acked(uint32_t noc, uint32_t inc = 1) {
  uint32_t stream_id = get_stream_index<risc_type>(noc);
  uint32_t val = NOC_STREAM_READ_REG(stream_id, STREAM_NONPOSTED_ATOMIC_RESP_RECEIVED) + inc;
  NOC_STREAM_WRITE_REG(stream_id, STREAM_NONPOSTED_ATOMIC_RESP_RECEIVED, val);
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void set_noc_nonposted_atomics_acked(uint32_t noc, uint32_t val) {
  NOC_STREAM_WRITE_REG(get_stream_index<risc_type>(noc), STREAM_NONPOSTED_ATOMIC_RESP_RECEIVED, val);
}

// noc_posted_writes_num_issued
template<uint32_t risc_type>
inline __attribute__((always_inline)) uint32_t get_noc_posted_writes_num_issued(uint32_t noc) {
  return NOC_STREAM_READ_REG(get_stream_index<risc_type>(noc), STREAM_POSTED_WR_REQ_SENT);
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void inc_noc_posted_writes_num_issued(uint32_t noc, uint32_t inc = 1) {
  uint32_t stream_id = get_stream_index<risc_type>(noc);
  uint32_t val = NOC_STREAM_READ_REG(stream_id, STREAM_POSTED_WR_REQ_SENT) + inc;
  NOC_STREAM_WRITE_REG(stream_id, STREAM_POSTED_WR_REQ_SENT, val);
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void set_noc_posted_writes_num_issued(uint32_t noc, uint32_t val) {
  NOC_STREAM_WRITE_REG(get_stream_index<risc_type>(noc), STREAM_POSTED_WR_REQ_SENT, val);
}

#endif
