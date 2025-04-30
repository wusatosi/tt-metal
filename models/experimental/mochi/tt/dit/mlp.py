import ttnn
from models.common.lightweightmodule import LightweightModule
import torch
from models.experimental.mochi.tt.common import matmul_2d_config, ff_matmul_config

from functools import partial


class FeedForward(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        in_features: int,
        hidden_size: int,
        multiple_of: int,
        ffn_dim_multiplier: float = None,
        state_dict_prefix=None,
        seq_shard: bool = False,
    ):
        super().__init__()
        # assert len(mesh_device.get_devices()) == 1, "Only single-device inference is supported for feedforward layers"

        # Calculate hidden size according to Mochi specs
        hidden_size = int(2 * hidden_size / 3)
        if ffn_dim_multiplier is not None:
            hidden_size = int(ffn_dim_multiplier * hidden_size)
        hidden_size = multiple_of * ((hidden_size + multiple_of - 1) // multiple_of)

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        torch_weight = lambda name: torch.transpose(self.state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)

        cache_name = lambda name: weight_cache_path / (state_dict_prefix + f".{name}")

        # TODO: Handle swizzling data when fracturing w1 on columns
        self.seq_shard = seq_shard
        as_tensor = lambda name, pt_tensor, type, dim: ttnn.as_tensor(
            pt_tensor,
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=dim),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name),
        )
        fp32_dest_acc = True
        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc,
            packer_l1_acc=True,
        )
        # Sharded weights
        # Split w1 and w3 into two separate tensors
        w1_tensor, w3_tensor = torch_weight("w1").chunk(2, dim=-1)
        self.w1 = ttnn.unsqueeze_to_4D(as_tensor("w1", w1_tensor, ttnn.bfloat16, dim=-1))
        self.w3 = ttnn.unsqueeze_to_4D(as_tensor("w3", w3_tensor, ttnn.bfloat16, dim=-1))
        self.w2 = ttnn.unsqueeze_to_4D(as_tensor("w2", torch_weight("w2"), ttnn.bfloat16, dim=-1))

        if seq_shard:
            self.w13_config = partial(
                ff_matmul_config,
                k=in_features,
                n=hidden_size,
                in0_block_w=6,
                num_out_blocks_h=2,
                num_out_blocks_w=4,
                grid_size=(8, 8),
                fp32_dest_acc_en=fp32_dest_acc,
            )
            self.w2_config = partial(
                ff_matmul_config,
                k=hidden_size,
                n=in_features,
                in0_block_w=8,
                num_out_blocks_h=2,
                num_out_blocks_w=2,
                grid_size=(8, 8),
                fp32_dest_acc_en=fp32_dest_acc,
            )
        else:
            self.w13_config = partial(
                matmul_2d_config,
                k=in_features,
                n=hidden_size // self.num_devices,
                grid_size=(8, 8),
            )
            self.w2_config = partial(
                matmul_2d_config,
                k=hidden_size,
                n=in_features // self.num_devices,
                grid_size=(8, 8),
            )

    def dealloc(self):
        ttnn.deallocate(self.w1)
        ttnn.deallocate(self.w2)
        ttnn.deallocate(self.w3)

    def forward(
        self,
        x_1BSD: ttnn.Tensor,
        ccl_semaphore_handles: dict,
        worker_sub_device_id: ttnn.SubDeviceId,
        topology: ttnn.Topology,
    ) -> ttnn.Tensor:
        B = x_1BSD.shape[1]
        S = x_1BSD.shape[2]
        D = x_1BSD.shape[3]
        assert B == 1, "Batch size must be 1, got {}".format(B)

        # W1 computation (includes both x and gate paths)

        if self.seq_shard:
            w1 = ttnn.experimental.all_gather_async(
                self.w1,
                dim=3,
                multi_device_global_semaphore=ccl_semaphore_handles["w1"],
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
                subdevice_id=worker_sub_device_id,
            )
        else:
            w1 = self.w1
        w1_out_1BSF = ttnn.linear(
            x_1BSD,
            w1,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat16,
            memory_config=x_1BSD.memory_config(),
            program_config=self.w13_config(m=S),
        )

        if self.seq_shard:
            w3 = ttnn.experimental.all_gather_async(
                self.w3,
                dim=3,
                multi_device_global_semaphore=ccl_semaphore_handles["w3"],
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
                subdevice_id=worker_sub_device_id,
            )
        else:
            w3 = self.w3
        w3_out_1BSF = ttnn.linear(
            x_1BSD,
            w3,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat16,
            memory_config=x_1BSD.memory_config(),
            program_config=self.w13_config(m=S),
        )

        # Apply SiLU and multiply with gate
        w2_in_1BSF = ttnn.multiply(
            w1_out_1BSF,
            w3_out_1BSF,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=w1_out_1BSF.memory_config(),
        )

        # W2 computation
        if self.seq_shard:
            w2 = ttnn.experimental.all_gather_async(
                self.w2,
                dim=3,
                multi_device_global_semaphore=ccl_semaphore_handles["w2"],
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
                subdevice_id=worker_sub_device_id,
            )
        else:
            w2 = self.w2
            w2_in_1BSF = ttnn.experimental.all_gather_async(
                w2_in_1BSF,
                dim=3,
                multi_device_global_semaphore=ccl_semaphore_handles["w2_in"],
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
                subdevice_id=worker_sub_device_id,
            )
        result_1BSD = ttnn.linear(
            w2_in_1BSF,
            w2,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat16,
            memory_config=w2_in_1BSF.memory_config(),
            program_config=self.w2_config(m=S),
        )

        # Necessary to infuse padded shape into output (deleted by all_gather_async)
        result_1BSD = ttnn.reshape(
            result_1BSD, (1, B, S, D // (1 if self.seq_shard else self.num_devices)), result_1BSD.shape
        )
        return result_1BSD
