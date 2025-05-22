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
        self.experts_per_tok = args.experts_per_tok
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
        self.num_experts = torch_weight("gate").shape[-1]
        self.experts = OffloadedMLP(
            self.num_experts,
            self.mesh_device,
            args,
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
        routing_weights, selected_experts = torch.topk(routing_weights, self.experts_per_tok, dim=-1)
        # only diff with mixtral sparse moe block!
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        output_rows = []
        # We have to run the experts one row at a time in prefill
        # TODO: group tokens by expert so we run at most one forward per expert
        for row in range(x.shape[2]):
            x_row = x[:, :, row : row + 1, :]
            experts_row = selected_experts[0, 0, row]
            # Run each expert in turn, store result before using to allow for lazy execution
            outputs = [self.experts.forward(index.item(), x_row, mode) for index in experts_row]

            # Multiply the output tensors by the routing weight
            for i, output_tensor in enumerate(outputs):
                output_tensor *= routing_weights[0, 0, row, i].item()

            # Sum all the outputs
            output_rows.append(sum(outputs))

        result = ttnn.stack(output_rows, dim=2)
        return result
