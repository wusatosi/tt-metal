# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import math
from loguru import logger

from models.common.lightweightmodule import LightweightModule
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
)

from tests.ttnn.unit_tests.operations.prefetcher_common import (
    get_core_ranges,
)
from tests.tt_eager.python_api_testing.unit_testing.misc.test_matmul_1d_gather_in0 import (
    run_multi_core_matmul_1d,
    PREFETCHER_NOC1_GRID,
    PREFETCHER_HOP_GRID,
)
from models.perf.benchmarking_utils import BenchmarkProfiler


class LlamaPrefetcher(LightweightModule):
    def __init__(self, mesh_device, n_tensors, n_layers, init_ccl=False):
        logger.info("Running LlamaPrefetcher")

        self.mesh_device = mesh_device
        self.n_tensors = n_tensors
        self.n_layers = n_layers

        ###### Set up GlobalCB ######
        num_reader_cores = 12
        num_global_cb_receivers = 2

        (
            self.dram_cores,
            self.sender_cores,
            self.receiver_cores_list,
            self.receiver_cores,
            self.worker_cores_range_set,
            self.mm_optimised_ring_cores,
            self.hop_grid,
        ) = get_core_ranges(num_reader_cores, num_global_cb_receivers, is_functional_test=False)

        max_tile_size = 1088
        self.global_cb_size = 750 * max_tile_size
        self.sender_receiver_mapping = list(zip(self.sender_cores, self.receiver_cores))
        self.global_circular_buffer = ttnn.create_global_circular_buffer(
            self.mesh_device, self.sender_receiver_mapping, self.global_cb_size
        )
        logger.info(f"GlobalCB size {self.global_cb_size}")

        ##### Set up the input tensors #####
        self.dram_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(core_coord, core_coord) for core_coord in self.dram_cores]
        )
        self.sender_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(core_coord, core_coord) for core_coord in self.sender_cores]
        )

        ##### Setup up sub devices #####
        self.prefetcher_sub_device = ttnn.SubDevice([self.sender_core_range_set])
        self.worker_sub_device = ttnn.SubDevice([self.worker_cores_range_set])

        if init_ccl:
            self.sub_device_manager = create_and_load_sub_device_manager_with_fabric_interface(
                mesh_device, [self.prefetcher_sub_device, self.worker_sub_device], 1, 0, True
            )
        else:
            self.sub_device_manager = mesh_device.create_sub_device_manager(
                [self.prefetcher_sub_device, self.worker_sub_device], 0
            )
            mesh_device.load_sub_device_manager(self.sub_device_manager)
        self.prefetcher_sub_device_id = ttnn.SubDeviceId(0)
        self.worker_sub_device_id = ttnn.SubDeviceId(1)

        self.tensors = []
        self.tensor_addrs = []  # List of buffer addresses

        ##### Create the tensors to be prefetched #####
        K = 2048
        N = 12 * 32 * 8  # Nice divisible number
        tensor_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                self.dram_core_range_set,
                [K, N // len(self.dram_cores)],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        tensor = ttnn.from_torch(
            torch.ones(1, 1, K, N),
            device=self.mesh_device,
            dtype=ttnn.bfloat8_b,
            memory_config=tensor_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.TILE_LAYOUT,
        )

        for _ in range(self.n_tensors * self.n_layers):
            self.insert_tensor(tensor)

        self.input_tensors = self.get_input_tensors()

    def buffer_address(self, tensor):
        addr = []
        for i, ten in enumerate(ttnn.get_device_tensors(tensor)):
            addr.append(ten.buffer_address())
            if len(addr) > 0:
                assert addr[i - 1] == addr[i], f"Expected {addr[i-1]} == {addr[i]}"
        return addr[0]

    def insert_tensor(self, tensor: ttnn.Tensor):
        self.tensors.append(tensor)
        self.tensor_addrs.append(self.buffer_address(tensor))

    def get_tensor_addrs(self):
        assert (
            len(self.tensor_addrs) == self.n_tensors * self.n_layers
        ), f"Expected {self.n_tensors * self.n_layers} tensor addresses, got {len(self.tensor_addrs)}"

        tensor_addrs = torch.tensor(self.tensor_addrs)
        tensor_addrs = tensor_addrs.repeat(len(self.dram_cores), 1)
        tensor_addrs_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self.sender_core_range_set,
                [tensor_addrs.shape[0] // len(self.dram_cores), tensor_addrs.shape[1]],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        tt_tensor_addrs = ttnn.as_tensor(
            tensor_addrs,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            memory_config=tensor_addrs_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        return tt_tensor_addrs

    def get_input_tensors(self):
        assert (
            len(self.tensors) >= self.n_tensors
        ), f"Expected at least {self.n_tensors} tensors, got {len(self.tensors)}"

        return self.tensors[: self.n_tensors] + [self.get_tensor_addrs()]

    def run_op(self):
        ttnn.dram_prefetcher(
            self.input_tensors,
            self.n_layers,
            self.global_circular_buffer,
            non_blocking=True,
        )
        self.mesh_device.set_sub_device_stall_group([self.worker_sub_device_id])


class MatmulOp(LightweightModule):
    def __init__(
        self,
        mesh_device,
        in0_dtype,
        in1_dtype,
        fidelity,
        fp32_acc_mode,
        packer_l1_acc,
        B,
        M,
        K,
        N,
    ):
        logger.info("Running MatmulOp")

        self.mesh_device = mesh_device

        self.setup = run_multi_core_matmul_1d(
            mesh_device,
            in0_dtype,
            in1_dtype,
            fidelity,
            False,  # has_bias
            fp32_acc_mode,
            packer_l1_acc,
            B,
            M,
            K,
            N,
            activation=None,
            grid=PREFETCHER_NOC1_GRID,
            use_arbitrary_cores=True,
            use_physical_to_logical_mapping=False,
            hop_grid=PREFETCHER_HOP_GRID,
            return_setup=True,
        )

    def run_op(self):
        out = ttnn.matmul(
            self.setup["in0_tt"],
            self.setup["in1_tt"],
            program_config=self.setup["program_config"],
            memory_config=self.setup["memory_config"],
            compute_kernel_config=self.setup["compute_kernel_config"],
        )

        return out
