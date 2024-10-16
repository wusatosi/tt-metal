// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifndef CKERNEL_PERF_COUNTERS_H
#define CKERNEL_PERF_COUNTERS_H

#include "tensix.h"

/**
 * @file kernel_perf_counters.h
 * @brief Provides template-based performance counter classes for hardware performance counters.
 *
 * Defines classes to configure and read hardware performance counters for components such as the FPU, L1 cache,
 * instruction threads, TDMA Pack, and TDMA Unpack. Each class offers methods to initialize, start,
 * stop, and read counters specific to its module.
 *
 * Example:
 *
   // Initialize the FPU performance counter in continuous mode
   MATH(FPU_PerformanceCounter::init(PerfCounterMode::Continuous));

   // Select the FPU counter to monitor
   MATH(FPU_PerformanceCounter::select_counter(FPU_PerformanceCounter::Counter::FPU));

   // Start the performance counter
   MATH(FPU_PerformanceCounter::start());

   // Perform operations to be measured
   // ... (your code here)

   // Stop the performance counter
   MATH(FPU_PerformanceCounter::stop());

   // Read the cycle count and counter value
   MATH(uint32_t cycles = FPU_PerformanceCounter::read_cycle_count());
   MATH(uint32_t value = FPU_PerformanceCounter::read_counter_value());

   // Or using predefined API
   init_fpu_perf_counters();
   start_fpu_perf_counters();
   ...
   stop_fpu_perf_counters();
   uint32_t fpu_value = get_fpu_perf_counter_value();

 * Note:
 *    - If you want to call read_counter_value() immediately after select_counter, you need to add wait(1).
 *    - L1 counters are using multiplexer. If multiplexer bit is changed you need to reset counters(stop/start)
 */

namespace ckernel {

namespace perf {

enum PerfCounterMode : uint8_t {
   Continuous = 0, // Continuous mode, controlled by start/stop
   AutoStop   = 1, // Auto-stop after reference period
};

enum PerfCounterSignal : uint8_t {
   Request = 0,
   Grant   = 1,
};

// Define Start/Stop Bits
constexpr uint32_t START_BIT = (1 << 0);
constexpr uint32_t STOP_BIT  = (1 << 1);
constexpr uint32_t ZERO      = 0;

union PerfCounterConfig {
    uint32_t value;  // Full 32-bit register access

    struct {
        uint32_t mode               : 1;  // Bits 0  - Operation Mode Continuous/Auto-stop
        uint32_t reserved1          : 7;  // Bits 7:1
        uint32_t counter_id         : 8;  // Bits 15:8 - Counter Selection
        uint32_t signal_type        : 1;  // Bit 16 - Request/Grant Selection
        uint32_t reserved2          : 15; // Bits 31:17
    } fields;
};


// Template class for Performance Counter
template <typename Derived>
class PerformanceCounterBase {
private:
   static inline uint32_t get_reg_value(uintptr_t reg_addr) {
      return *reinterpret_cast<volatile uint32_t*>(reg_addr);
   }

   static inline void set_reg_value(uintptr_t reg_addr, uint32_t value) {
      *reinterpret_cast<volatile uint32_t*>(reg_addr) = value;
   }

protected:
   // Access derived class's static members
   static constexpr uintptr_t reference_period_reg_addr = Derived::reference_period_reg_addr;
   static constexpr uintptr_t config_reg_addr           = Derived::config_reg_addr;
   static constexpr uintptr_t start_stop_reg_addr       = Derived::start_stop_reg_addr;
   static constexpr uintptr_t cycle_count_reg_addr      = Derived::cycle_count_reg_addr;
   static constexpr uintptr_t counter_value_reg_addr    = Derived::counter_value_reg_addr;

   static inline uint32_t get_config() {
      return get_reg_value(config_reg_addr);
   }

   static inline void set_config(uint32_t value) {
      set_reg_value(config_reg_addr, value);
   }

   static inline uint32_t get_period() {
      return get_reg_value(reference_period_reg_addr);
   }

   static inline void set_period(uint32_t value = 0xFFFFFFFF) {
      set_reg_value(reference_period_reg_addr, value);
   }

   static inline uint32_t get_start_stop() {
      return get_reg_value(start_stop_reg_addr);
   }

   static inline void set_start_stop(uint32_t value) {
      set_reg_value(start_stop_reg_addr, value);
   }

   static inline void select_counter_base(uint8_t counter_id, PerfCounterSignal signal_type = PerfCounterSignal::Request) {
      PerfCounterConfig config;
      config.value = get_config();
      config.fields.counter_id = counter_id & 0xFF;
      config.fields.signal_type = signal_type;
      set_config(config.value);
   }

public:
   // Initialize the performance counter
   static inline void init(PerfCounterMode mode = PerfCounterMode::Continuous, uint32_t period = 0xFFFFFFFF) {
         PerfCounterConfig config;

         config.value = 0;
         config.fields.mode = mode & 0x1;
         set_config(config.value);

         set_period(period);
   }

   // Start the performance counter
   static inline void start() {
      set_start_stop(ZERO);
      set_start_stop(START_BIT);
   }

   // Stop the performance counter
   static inline void stop() {
      set_start_stop(ZERO);
      set_start_stop(STOP_BIT);
   }

  // Read the cycle count
   static inline uint32_t read_cycle_count() {
      return get_reg_value(cycle_count_reg_addr);
   }

    // Read the counter value
   static inline uint32_t read_counter_value() {
      return get_reg_value(counter_value_reg_addr);
   }

};

// Derived class for FPU Performance Counter
class FPU_PerformanceCounter : public PerformanceCounterBase<FPU_PerformanceCounter> {
public:
   // Define the register addresses specific to FPU
   static constexpr uintptr_t reference_period_reg_addr = RISCV_DEBUG_REG_PERF_CNT_FPU0;
   static constexpr uintptr_t config_reg_addr           = RISCV_DEBUG_REG_PERF_CNT_FPU1;
   static constexpr uintptr_t start_stop_reg_addr       = RISCV_DEBUG_REG_PERF_CNT_FPU2;
   static constexpr uintptr_t cycle_count_reg_addr      = RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU;
   static constexpr uintptr_t counter_value_reg_addr    = RISCV_DEBUG_REG_PERF_CNT_OUT_H_FPU;

   // Enum class for FPU counters
   enum class Counter : uint8_t {
      FPU      = 0,
      SFPU     = 1,
      COUNT    = 2,
   };

   // Public method to select counter, specific to FPU
   static inline void select_counter(Counter counter, PerfCounterSignal signal = PerfCounterSignal::Request) {
      // Call the base class's select_counter method
      select_counter_base(static_cast<uint8_t>(counter), signal);
   }
};

// Derived class for L1 Performance Counter
class L1_PerformanceCounter : public PerformanceCounterBase<L1_PerformanceCounter> {
public:
   // Define the register addresses specific to FPU
   static constexpr uintptr_t reference_period_reg_addr = RISCV_DEBUG_REG_PERF_CNT_L1_0;
   static constexpr uintptr_t config_reg_addr           = RISCV_DEBUG_REG_PERF_CNT_L1_1;
   static constexpr uintptr_t start_stop_reg_addr       = RISCV_DEBUG_REG_PERF_CNT_L1_2;
   static constexpr uintptr_t cycle_count_reg_addr      = RISCV_DEBUG_REG_PERF_CNT_OUT_L_L1;
   static constexpr uintptr_t counter_value_reg_addr    = RISCV_DEBUG_REG_PERF_CNT_OUT_H_L1;
   // mux register used to select specific group of counters
   static constexpr uintptr_t mux_ctrl_reg_addr         = RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL;
   // Enum class for FPU counters
   enum class Counter : uint8_t {
      // Counters when MUX_CTRL bit 4 = 0
      // Mode(15:8) values range from 0 to 7
      RING0_NIU_L1_INCOMING1             = 0, // Mode(15:8) = 0, MUX_CTRL bit 4 = 0
      RING0_NIU_L1_INCOMING0             = 1, // Mode(15:8) = 1, MUX_CTRL bit 4 = 0
      RING0_NIU_L1_OUTGOING1             = 2, // Mode(15:8) = 2, MUX_CTRL bit 4 = 0
      RING0_NIU_L1_OUTGOING0             = 3, // Mode(15:8) = 3, MUX_CTRL bit 4 = 0
      L1_ARBITRATION_TDMA_BUNDLE_1       = 4, // Mode(15:8) = 4, MUX_CTRL bit 4 = 0
      L1_ARBITRATION_TDMA_BUNDLE_0       = 5, // Mode(15:8) = 5, MUX_CTRL bit 4 = 0
      L1_ARBITRATION_UNPACKER_ECC_PACKER = 6, // Mode(15:8) = 6, MUX_CTRL bit 4 = 0
      L1_NO_ARBITRATION_UNPACKER_0       = 7, // Mode(15:8) = 7, MUX_CTRL bit 4 = 0
      // Counters when MUX_CTRL bit 4 = 1
      // Mode(15:8) values range from 0 to 7
      RING1_NIU_L1_INCOMING1             = 8, // Mode(15:8) = 0, MUX_CTRL bit 4 = 1
      RING1_NIU_L1_INCOMING0             = 9, // Mode(15:8) = 1, MUX_CTRL bit 4 = 1
      RING1_NIU_L1_OUTGOING1             = 10, // Mode(15:8) = 2, MUX_CTRL bit 4 = 1
      RING1_NIU_L1_OUTGOING0             = 11, // Mode(15:8) = 3, MUX_CTRL bit 4 = 1
      TDMA_EXT_UNPACKER_INTERFACE_1      = 12, // Mode(15:8) = 4, MUX_CTRL bit 4 = 1
      TDMA_EXT_UNPACKER_INTERFACE_2      = 13, // Mode(15:8) = 5, MUX_CTRL bit 4 = 1
      TDMA_EXT_UNPACKER_INTERFACE_3      = 14, // Mode(15:8) = 6, MUX_CTRL bit 4 = 1
      TDMA_PACKER_2_WRITE_INTERFACE      = 15, // Mode(15:8) = 7, MUX_CTRL bit 4 = 1
      COUNT                              = 16,
   };

   // Public method to select counter, specific to FPU
   static inline void select_counter(Counter counter, PerfCounterSignal signal = PerfCounterSignal::Request) {
      uint8_t counter_id = static_cast<uint8_t>(counter);
      set_mux_ctrl(counter_id >= static_cast<uint8_t>(Counter::RING1_NIU_L1_INCOMING1));

      // Call the base class's select_counter method
      select_counter_base(counter_id & 0x7, signal);
   }
private:
   // Function to set the MUX_CTRL bit 4
   // DEVNOTE: if mux ctrl is changed reset counter
   static inline void set_mux_ctrl(bool bit4_value) {
      if (bit4_value) {
         *reinterpret_cast<volatile uint32_t*>(mux_ctrl_reg_addr) |= (1 << 4);
      } else {
         *reinterpret_cast<volatile uint32_t*>(mux_ctrl_reg_addr) &= 0xFFFFFFEF;
      }
   }
};

class InstrnThread_PerformanceCounter : public PerformanceCounterBase<InstrnThread_PerformanceCounter> {
public:
    // Register addresses specific to Instruction Thread
    static constexpr uintptr_t reference_period_reg_addr = RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0;
    static constexpr uintptr_t config_reg_addr           = RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD1;
    static constexpr uintptr_t start_stop_reg_addr       = RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD2;
    static constexpr uintptr_t cycle_count_reg_addr      = RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD;
    static constexpr uintptr_t counter_value_reg_addr    = RISCV_DEBUG_REG_PERF_CNT_OUT_H_INSTRN_THREAD;

    // Enum class for Instruction Thread counters
    enum class Counter : uint8_t {
        INST_CFG                     = 0,  // Mode(15:8) = 0
        INST_SYNC                    = 1,  // Mode(15:8) = 1
        INST_THCON                   = 2,  // Mode(15:8) = 2
        INST_XSEARCH                 = 3,  // Mode(15:8) = 3
        INST_MOVE                    = 4,  // Mode(15:8) = 4
        INST_MATH                    = 5,  // Mode(15:8) = 5
        INST_UNPACK                  = 6,  // Mode(15:8) = 6
        INST_PACK                    = 7,  // Mode(15:8) = 7
        STALLED                      = 8,  // Mode(15:8) = 8
        STALL_RSN_SRCA_CLEARED_0     = 9,  // Mode(15:8) = 9
        STALL_RSN_SRCA_CLEARED_1     = 10, // Mode(15:8) = 10
        STALL_RSN_SRCA_CLEARED_2     = 11, // Mode(15:8) = 11
        STALL_RSN_SRCB_CLEARED_0     = 12, // Mode(15:8) = 12
        STALL_RSN_SRCB_CLEARED_1     = 13, // Mode(15:8) = 13
        STALL_RSN_SRCB_CLEARED_2     = 14, // Mode(15:8) = 14
        STALL_RSN_SRCA_VALID_0       = 15, // Mode(15:8) = 15
        STALL_RSN_SRCA_VALID_1       = 16, // Mode(15:8) = 16
        STALL_RSN_SRCA_VALID_2       = 17, // Mode(15:8) = 17
        STALL_RSN_SRCB_VALID_0       = 18, // Mode(15:8) = 18
        STALL_RSN_SRCB_VALID_1       = 19, // Mode(15:8) = 19
        STALL_RSN_SRCB_VALID_2       = 20, // Mode(15:8) = 20
        STALL_RSN_THCON              = 21, // Mode(15:8) = 21
        STALL_RSN_PACK0              = 22, // Mode(15:8) = 22
        STALL_RSN_MATH               = 23, // Mode(15:8) = 23
        RSN_SEM_ZERO                 = 24, // Mode(15:8) = 24
        RSN_SEM_MAX                  = 25, // Mode(15:8) = 25
        STALL_RSN_MOVE               = 26, // Mode(15:8) = 26
        STALL_RSN_TRISC_REG_ACCESS   = 27, // Mode(15:8) = 27
        STALL_RSN_SFPU               = 28, // Mode(15:8) = 28
        COUNT                        = 29,
    };

    // Public method to select counter
    static inline void select_counter(Counter counter, PerfCounterSignal signal = PerfCounterSignal::Request) {
        uint8_t counter_id = static_cast<uint8_t>(counter);
        select_counter_base(counter_id, signal);
    }
};

class TDMAPack_PerformanceCounter : public PerformanceCounterBase<TDMAPack_PerformanceCounter> {
public:
    // Register addresses specific to TDMA Pack
    static constexpr uintptr_t reference_period_reg_addr = RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK0;
    static constexpr uintptr_t config_reg_addr           = RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK1;
    static constexpr uintptr_t start_stop_reg_addr       = RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK2;
    static constexpr uintptr_t cycle_count_reg_addr      = RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_PACK;
    static constexpr uintptr_t counter_value_reg_addr    = RISCV_DEBUG_REG_PERF_CNT_OUT_H_TDMA_PACK;

    // Enum class for TDMA Pack counters
    enum class Counter : uint8_t {
        TDMA_DSTAC_REGIF_RDEN_RAW_0           = 0, // Mode(15:8) = 0
        TDMA_DSTAC_REGIF_RDEN_RAW_1           = 1, // Mode(15:8) = 1
        TDMA_DSTAC_REGIF_RDEN_RAW_2           = 2, // Mode(15:8) = 2
        TDMA_DSTAC_REGIF_RDEN_RAW_3           = 3, // Mode(15:8) = 3
        TDMA_PACK_BUSY_0                      = 4, // Mode(15:8) = 4
        TDMA_PACK_BUSY_1                      = 5, // Mode(15:8) = 5
        TDMA_PACK_BUSY_2                      = 6, // Mode(15:8) = 6
        TDMA_PACK_BUSY_3                      = 7, // Mode(15:8) = 7
        COUNT                                 = 8,
    };

    // Public method to select counter
    static inline void select_counter(Counter counter, PerfCounterSignal signal = PerfCounterSignal::Request) {
        uint8_t counter_id = static_cast<uint8_t>(counter);
        select_counter_base(counter_id, signal);
    }
};

class TDMAUnpack_PerformanceCounter : public PerformanceCounterBase<TDMAUnpack_PerformanceCounter> {
public:
    // Register addresses specific to TDMA Unpack
    static constexpr uintptr_t reference_period_reg_addr = RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0;
    static constexpr uintptr_t config_reg_addr           = RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK1;
    static constexpr uintptr_t start_stop_reg_addr       = RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK2;
    static constexpr uintptr_t cycle_count_reg_addr      = RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK;
    static constexpr uintptr_t counter_value_reg_addr    = RISCV_DEBUG_REG_PERF_CNT_OUT_H_TDMA_UNPACK;

    // Enum class for TDMA Unpack counters
    enum class Counter : uint8_t {
        MATH_INSTRN_VALID_SRC_DATA_READY             = 0, // Mode(15:8) = 0
        MATH_INSTRN_VALID_DEST2SRC_POST_STALL        = 1, // Mode(15:8) = 1
        MATH_INSTRN_VALID_FIDELITY_PHASES_ONGOING    = 2, // Mode(15:8) = 2
        O_MATH_INSTRNBUF_RDEN                        = 3, // Mode(15:8) = 3
        MATH_INSTRN_VALID                            = 4, // Mode(15:8) = 4
        TDMA_SRCA_REGIF_WREN                         = 5, // Mode(15:8) = 5
        TDMA_SRCB_REGIF_WREN                         = 6, // Mode(15:8) = 6
        TDMA_UNPACK_BUSY_0                           = 7, // Mode(15:8) = 7
        TDMA_UNPACK_BUSY_1                           = 8, // Mode(15:8) = 8
        TDMA_UNPACK_BUSY_2                           = 9, // Mode(15:8) = 9
        TDMA_UNPACK_BUSY_3                           = 10,// Mode(15:8) = 10
        COUNT                                        = 11,
    };

    // Public method to select counter
    static inline void select_counter(Counter counter, PerfCounterSignal signal = PerfCounterSignal::Request) {
        uint8_t counter_id = static_cast<uint8_t>(counter);
        select_counter_base(counter_id, signal);
    }
};


// API for FPU perf counters
ALWI void init_fpu_perf_counters(PerfCounterMode mode = PerfCounterMode::Continuous, uint32_t period = 0xFFFFFFFF) {
   MATH({
      FPU_PerformanceCounter::init(mode, period);
   });
}

ALWI void start_fpu_perf_counters() {
   MATH({
      FPU_PerformanceCounter::start();
   });
}

ALWI void stop_fpu_perf_counters() {
   MATH({
      FPU_PerformanceCounter::stop();
   });
}

ALWI uint32_t get_fpu_perf_counter_value(
   FPU_PerformanceCounter::Counter counter = FPU_PerformanceCounter::Counter::FPU,
   PerfCounterSignal signal = PerfCounterSignal::Request) {
      MATH({
         FPU_PerformanceCounter::select_counter(counter, signal);
         wait(1);
         return FPU_PerformanceCounter::read_counter_value();
      });
      return 0;
}

ALWI uint32_t get_fpu_perf_counter_cycles() {
   MATH({
      return FPU_PerformanceCounter::read_cycle_count();
   });

   return 0;
}

} // namespace ckernel::perf
} // namespace ckernel
#endif
