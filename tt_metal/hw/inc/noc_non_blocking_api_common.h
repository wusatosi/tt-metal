// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _NOC_NON_BLOCKING_API_COMMON_H_
#define _NOC_NON_BLOCKING_API_COMMON_H_

#include <stdint.h>

#include <cstdint>

#include "noc_overlay_parameters.h"

#define   STREAM_RD_RESP_RECEIVED   0
#define   STREAM_NONPOSTED_WR_REQ_SENT   1
#define   STREAM_NONPOSTED_WR_ACK_RECEIVED   2
#define   STREAM_NONPOSTED_ATOMIC_RESP_RECEIVED   3
#define   STREAM_POSTED_WR_REQ_SENT   4

const uint32_t OPERAND_BRISC_STREAM = 0;
const uint32_t OPERAND_NCRISC_STREAM = 1;

inline __attribute__((always_inline)) uint32_t get_stream_reg_index(uint32_t noc, uint32_t index) {
  return uint32_t(noc << 3) + index;
}

// noc_reads_num_issued
template<uint32_t risc_type>
inline __attribute__((always_inline)) uint32_t get_noc_reads_num_issued(uint32_t noc) {
  if constexpr (risc_type == static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0)) {
    return NOC_STREAM_READ_REG(OPERAND_BRISC_STREAM, get_stream_reg_index(noc, STREAM_RD_RESP_RECEIVED));
  } else {
    return NOC_STREAM_READ_REG(OPERAND_NCRISC_STREAM, get_stream_reg_index(noc, STREAM_RD_RESP_RECEIVED));
  }
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void inc_noc_reads_num_issued(uint32_t noc, uint32_t inc = 1) {
  uint32_t id = get_stream_reg_index(noc, STREAM_RD_RESP_RECEIVED);
  if constexpr (risc_type == static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0)) {
    volatile uint32_t val = NOC_STREAM_READ_REG(OPERAND_BRISC_STREAM, id) + inc;
    NOC_STREAM_WRITE_REG(OPERAND_BRISC_STREAM, id, val);
  } else {
    volatile uint32_t val = NOC_STREAM_READ_REG(OPERAND_NCRISC_STREAM, id) + inc;
    NOC_STREAM_WRITE_REG(OPERAND_NCRISC_STREAM, id, val);
  }
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void set_noc_reads_num_issued(uint32_t noc, uint32_t val) {
  if constexpr (risc_type == static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0)) {
    NOC_STREAM_WRITE_REG(OPERAND_BRISC_STREAM, get_stream_reg_index(noc, STREAM_RD_RESP_RECEIVED), val);
  } else {
    NOC_STREAM_WRITE_REG(OPERAND_NCRISC_STREAM, get_stream_reg_index(noc, STREAM_RD_RESP_RECEIVED), val);
  }
}

// noc_nonposted_writes_num_issued
template<uint32_t risc_type>
inline __attribute__((always_inline)) volatile uint32_t get_noc_nonposted_writes_num_issued(uint32_t noc) {
  if constexpr (risc_type == static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0)) {
    return NOC_STREAM_READ_REG(OPERAND_BRISC_STREAM, get_stream_reg_index(noc, STREAM_NONPOSTED_WR_REQ_SENT));
  } else {
    return NOC_STREAM_READ_REG(OPERAND_NCRISC_STREAM, get_stream_reg_index(noc, STREAM_NONPOSTED_WR_REQ_SENT));
  }
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void inc_noc_nonposted_writes_num_issued(uint32_t noc, uint32_t inc = 1) {
  uint32_t id = get_stream_reg_index(noc, STREAM_NONPOSTED_WR_REQ_SENT);
  if constexpr (risc_type == static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0)) {
    volatile uint32_t val = NOC_STREAM_READ_REG(OPERAND_BRISC_STREAM, id) + inc;
    NOC_STREAM_WRITE_REG(OPERAND_BRISC_STREAM, id, val);
  } else {
    volatile uint32_t val = NOC_STREAM_READ_REG(OPERAND_NCRISC_STREAM, id) + inc;
    NOC_STREAM_WRITE_REG(OPERAND_NCRISC_STREAM, id, val);
  }
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void set_noc_nonposted_writes_num_issued(int32_t noc, uint32_t val) {
  if constexpr (risc_type == static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0)) {
    NOC_STREAM_WRITE_REG(OPERAND_BRISC_STREAM, get_stream_reg_index(noc, STREAM_NONPOSTED_WR_REQ_SENT), val);
  } else {
    NOC_STREAM_WRITE_REG(OPERAND_NCRISC_STREAM, get_stream_reg_index(noc, STREAM_NONPOSTED_WR_REQ_SENT), val);
  }
}

// noc_nonposted_writes_acked
template<uint32_t risc_type>
inline __attribute__((always_inline)) uint32_t get_noc_nonposted_writes_acked(uint32_t noc) {
  if constexpr (risc_type == static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0)) {
    return NOC_STREAM_READ_REG(OPERAND_BRISC_STREAM, get_stream_reg_index(noc, STREAM_NONPOSTED_WR_ACK_RECEIVED));
  } else {
    return NOC_STREAM_READ_REG(OPERAND_NCRISC_STREAM, get_stream_reg_index(noc, STREAM_NONPOSTED_WR_ACK_RECEIVED));
  }
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void inc_noc_nonposted_writes_acked(uint32_t noc, uint32_t inc = 1) {
  uint32_t id = get_stream_reg_index(noc, STREAM_NONPOSTED_WR_ACK_RECEIVED);
  if constexpr (risc_type == static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0)) {
    volatile uint32_t val = NOC_STREAM_READ_REG(OPERAND_BRISC_STREAM, id) + inc;
    NOC_STREAM_WRITE_REG(OPERAND_BRISC_STREAM, id, val);
  } else {
    volatile uint32_t val = NOC_STREAM_READ_REG(OPERAND_NCRISC_STREAM, id) + inc;
    NOC_STREAM_WRITE_REG(OPERAND_NCRISC_STREAM, id, val);
  }
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void set_noc_nonposted_writes_acked(uint32_t noc, uint32_t val) {
  if constexpr (risc_type == static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0)) {
    NOC_STREAM_WRITE_REG(OPERAND_BRISC_STREAM, get_stream_reg_index(noc, STREAM_NONPOSTED_WR_ACK_RECEIVED), val);
  } else {
    NOC_STREAM_WRITE_REG(OPERAND_NCRISC_STREAM, get_stream_reg_index(noc, STREAM_NONPOSTED_WR_ACK_RECEIVED), val);
  }
}

// noc_nonposted_atomics_acked
template<uint32_t risc_type>
inline __attribute__((always_inline)) uint32_t get_noc_nonposted_atomics_acked(uint32_t noc) {
  if constexpr (risc_type == static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0)) {
    return NOC_STREAM_READ_REG(OPERAND_BRISC_STREAM, get_stream_reg_index(noc, STREAM_NONPOSTED_ATOMIC_RESP_RECEIVED));
  } else {
    return NOC_STREAM_READ_REG(OPERAND_NCRISC_STREAM, get_stream_reg_index(noc, STREAM_NONPOSTED_ATOMIC_RESP_RECEIVED));
  }
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void inc_noc_nonposted_atomics_acked(uint32_t noc, uint32_t inc = 1) {
  uint32_t id = get_stream_reg_index(noc, STREAM_NONPOSTED_ATOMIC_RESP_RECEIVED);
  if constexpr (risc_type == static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0)) {
    volatile uint32_t val = NOC_STREAM_READ_REG(OPERAND_BRISC_STREAM, id) + inc;
    NOC_STREAM_WRITE_REG(OPERAND_BRISC_STREAM, id, val);
  } else {
    volatile uint32_t val = NOC_STREAM_READ_REG(OPERAND_NCRISC_STREAM, id) + inc;
    NOC_STREAM_WRITE_REG(OPERAND_NCRISC_STREAM, id, val);
  }
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void set_noc_nonposted_atomics_acked(uint32_t noc, uint32_t val) {
  if constexpr (risc_type == static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0)) {
    NOC_STREAM_WRITE_REG(OPERAND_BRISC_STREAM, get_stream_reg_index(noc, STREAM_NONPOSTED_ATOMIC_RESP_RECEIVED), val);
  } else {
    NOC_STREAM_WRITE_REG(OPERAND_NCRISC_STREAM, get_stream_reg_index(noc, STREAM_NONPOSTED_ATOMIC_RESP_RECEIVED), val);
  }
}

// noc_posted_writes_num_issued
template<uint32_t risc_type>
inline __attribute__((always_inline)) uint32_t get_noc_posted_writes_num_issued(uint32_t noc) {
  if constexpr (risc_type == static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0)) {
    return NOC_STREAM_READ_REG(OPERAND_BRISC_STREAM, get_stream_reg_index(noc, STREAM_POSTED_WR_REQ_SENT));
  } else {
    return NOC_STREAM_READ_REG(OPERAND_NCRISC_STREAM, get_stream_reg_index(noc, STREAM_POSTED_WR_REQ_SENT));
  }
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void inc_noc_posted_writes_num_issued(uint32_t noc, uint32_t inc = 1) {
  uint32_t id = get_stream_reg_index(noc, STREAM_POSTED_WR_REQ_SENT);
  if constexpr (risc_type == static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0)) {
    volatile uint32_t val = NOC_STREAM_READ_REG(OPERAND_BRISC_STREAM, id) + inc;
    NOC_STREAM_WRITE_REG(OPERAND_BRISC_STREAM, id, val);
  } else {
    volatile uint32_t val = NOC_STREAM_READ_REG(OPERAND_NCRISC_STREAM, id) + inc;
    NOC_STREAM_WRITE_REG(OPERAND_NCRISC_STREAM, id, val);
  }
}

template<uint32_t risc_type>
inline __attribute__((always_inline)) void set_noc_posted_writes_num_issued(uint32_t noc, uint32_t val) {
  if constexpr (risc_type == static_cast<std::underlying_type_t<TensixProcessorTypes>>(TensixProcessorTypes::DM0)) {
    NOC_STREAM_WRITE_REG(OPERAND_BRISC_STREAM, get_stream_reg_index(noc, STREAM_POSTED_WR_REQ_SENT), val);
  } else {
    NOC_STREAM_WRITE_REG(OPERAND_NCRISC_STREAM, get_stream_reg_index(noc, STREAM_POSTED_WR_REQ_SENT), val);
  }
}

#endif
