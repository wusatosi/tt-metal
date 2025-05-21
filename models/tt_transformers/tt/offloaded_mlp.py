# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.model_config import TensorGroup


class OffloadedMLP(MLP):
    def __init__(
        self,
        num_experts,
        mesh_device,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        model_config,
        state_dict_prefix=None,
    ):
        LightweightModule.__init__(self)  # Skip MLP ctor as it would try to load device weights from a state_dict

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.dim = args.dim
        self.model_config = model_config
        self.layer_num = layer_num
        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)
        torch_weight = lambda name: torch.transpose(self.state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)

        host_tensor = lambda name, type, dims: ttnn.as_tensor(
            torch_weight(name),
            dtype=type,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=None if args.dummy_weights else weight_cache_path / f"{state_dict_prefix}.{name}",
        )

        device_tensor = lambda name, type, dims: ttnn.as_tensor(
            torch_weight(name),
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=args.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ff1_3_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF1_FF3
        )
        ff2_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF2
        )

        w1_dims = (-1, -2) if args.is_galaxy else (-2, -1)
        w2_dims = (-2, -1) if args.is_galaxy else (-1, -2)

        self.w1 = device_tensor("experts.0.gate_proj", ff1_3_dtype, dims=w1_dims)
        self.w2 = device_tensor("experts.0.down_proj", ff2_dtype, dims=w2_dims)
        self.w3 = device_tensor("experts.0.up_proj", ff1_3_dtype, dims=w1_dims)

        self.host_w1 = [host_tensor(f"experts.{i}.gate_proj", ff1_3_dtype, dims=w1_dims) for i in range(num_experts)]
        self.host_w2 = [host_tensor(f"experts.{i}.down_proj", ff2_dtype, dims=w2_dims) for i in range(num_experts)]
        self.host_w3 = [host_tensor(f"experts.{i}.up_proj", ff1_3_dtype, dims=w1_dims) for i in range(num_experts)]

    def forward(self, expert_index: int, x: ttnn.Tensor, mode) -> ttnn.Tensor:
        ttnn.copy_host_to_device_tensor(self.host_w1[expert_index], self.w1)
        ttnn.copy_host_to_device_tensor(self.host_w2[expert_index], self.w2)
        ttnn.copy_host_to_device_tensor(self.host_w3[expert_index], self.w3)

        return super().forward(x, mode)
