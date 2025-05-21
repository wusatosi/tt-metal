# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.offloaded_mlp import OffloadedMLP


class MOEStream(LightweightModule):
    def __init__(
        self, mesh_device, args, state_dict, weight_cache_path, layer_num, dtype, model_config, state_dict_prefix=None
    ):
        """
        1. Subclass OffloadedMLP and create one of it - it creates arrays of its weights on host for all experts and one on-device tensor for each
        2. Create the gate_weight tensor on device
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.args = args
        self.model_config = model_config
        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)
        torch_weight = lambda name: torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)

        self.gate = ttnn.as_tensor(
            torch_weight("gate"),
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=(ttnn.DRAM_MEMORY_CONFIG),
            cache_file_name=None if args.dummy_weights else weight_cache_path / f"{state_dict_prefix}.gate",
        )
        num_experts = torch_weight("gate").shape[-1]
        self.experts = OffloadedMLP(
            num_experts,
            self.mesh_device,
            self.args,
            state_dict,
            weight_cache_path,
            layer_num,
            dtype,
            model_config,
            state_dict_prefix,
        )

    def forward(self, x: ttnn.Tensor, mode) -> ttnn.Tensor:
        """
        1. Run the routing matmul and move the output back to the host.
        2. Run a top_k and norm
        3. For each selected expert call OffLoadedMLP.forward(index) which will send the weights and run the expert and return the output tensor
        4. Multiply the output tensors by the routing weight
        5. Sum all the output tensors and return the result
        """

        # Run the routing matmul and move the output back to the host.
        routing_tt = ttnn.linear(x, self.gate)
        routing_torch = ttnn.to_torch(ttnn.get_device_tensors(routing_tt)[0])  # replicated on all devices

        # Run a top_k and norm
        routing_weights = torch.nn.functional.softmax(routing_torch, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, len(self.experts), dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # Run each expert in turn, store result before using to allow for lazy execution
        outputs = []
        for index in selected_experts:
            outputs.append(self.experts.forward(index, x, mode))

        # Multiply the output tensors by the routing weight
        for i, output_tensor in enumerate(outputs):
            output_tensor *= routing_weights[i].item()

        # Sum all the output tensors and return the result
        return sum(outputs)
