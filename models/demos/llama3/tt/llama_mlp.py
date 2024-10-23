# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3.tt.llama_ccl import tt_all_reduce, tt_sharded_all_reduce
from models.demos.t3000.llama2_70b.tt.llama_common import ShardTensor2dMesh, ConcatMesh2DToTensor


class TtLlamaMLP(LightweightModule):
    def __init__(
        self, mesh_device, args, state_dict, weight_cache_path, layer_num, dtype, model_config, state_dict_prefix=None
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.model_config = model_config
        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)
        torch_weight = lambda name: torch.transpose(self.state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
        TG = args.is_galaxy
        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (state_dict_prefix + f".{name}")

        w1_w3_mem_config = args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
        w2_mem_config = args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)

        # TODO Clean up this code. With sharding, we load the normal weights and then shard them
        as_sharded_tensor = lambda name, type, dim: ttnn.as_tensor(
            torch_weight(name[:2]),  # Grab only the wX part of the name
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ShardTensor2dMesh(self.mesh_device, dims=dim, cluster_shape=(4, 8))
            if TG
            else ttnn.ShardTensorToMesh(self.mesh_device, dim=dim),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if TG else w2_mem_config if "w2" in name else w1_w3_mem_config,
            cache_file_name=cache_name(name),
        )

        # Sharded weights
        if TG:
            w1_dim = (-2, -1)
            w2_dim = (-1, -2)
        else:
            w1_dim = -1
            w2_dim = -2

        self.w1 = as_sharded_tensor(
            "w1_sharded", ttnn.bfloat4_b if self.args.is_large_model else ttnn.bfloat8_b, dim=w1_dim
        )  # bfp4 normally ok here but sub .99 pcc for llama 3.1 weights
        self.w2 = as_sharded_tensor("w2_sharded", ttnn.bfloat8_b, dim=w2_dim)
        self.w3 = as_sharded_tensor(
            "w3_sharded", ttnn.bfloat4_b if self.args.is_large_model else ttnn.bfloat8_b, dim=w1_dim
        )

    def forward(self, x: ttnn.Tensor, mode) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        TG = self.args.is_galaxy
        if TG:
            self.mlp_config = self.model_config["mlp"][mode]

        if mode == "decode":  # Sharded config
            if TG:
                pc_1 = None
                pc_2 = None
                pc_3 = None
            else:
                pc_1 = self.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"]
                pc_2 = self.model_config["DECODE_MLP_W2_PRG_CONFIG"]
                pc_3 = self.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"]
        else:  # Update the program configs based for prefill
            if seq_len >= 1024:
                # Reshape input to to fit on device and parallelize computation
                x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])
            if TG:
                batch_dim = 1 if seq_len < 1024 else seq_len // 1024  # Find the division factor

                pc_1 = self.mlp_config["FF1_PROGCFG"](seq_len)
                pc_2 = self.mlp_config["FF2_PROGCFG"](seq_len)
                pc_3 = self.mlp_config["FF1_PROGCFG"](seq_len)
            else:
                if seq_len >= 1024:  # Too big to compute. Set different program configs based on seqlen
                    pc_1 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"]
                    pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG"]
                    pc_3 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"]
                else:
                    pc_1 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG_128"](seq_len)
                    pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG_128"](seq_len)
                    pc_3 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG_128"](seq_len)

        # In decode mode (seqlen <= 32) do DRAM sharded matmuls
        # These use HiFi2; this drops 1 bit of the activations but would be FLOP-bound on 12 cores with HiFi4
        w1_out = ttnn.linear(
            x,
            self.w1,
            compute_kernel_config=self.args.compute_kernel_config_hifi2,
            core_grid=ttnn.CoreGrid(y=1, x=8) if not pc_1 else None,
            dtype=ttnn.bfloat16,
            program_config=pc_1,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
        )

        w3_out = ttnn.linear(
            x,
            self.w3,
            compute_kernel_config=self.args.compute_kernel_config_hifi2,
            core_grid=ttnn.CoreGrid(y=1, x=8) if not pc_3 else None,
            dtype=ttnn.bfloat16,
            program_config=pc_3,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(x)

        if TG:
            if mode == "decode":
                w1_out = tt_sharded_all_reduce(
                    w1_out,
                    self.mesh_device,
                    cluster_axis=1,
                    num_links=2,
                    memory_config=self.mlp_config["FF1_OUT_GATHERED_MEMCFG"],
                )
                w3_out = tt_sharded_all_reduce(
                    w3_out,
                    self.mesh_device,
                    cluster_axis=1,
                    num_links=2,
                    memory_config=self.mlp_config["FF1_OUT_GATHERED_MEMCFG"],
                )

                w1_out = ttnn.to_memory_config(w1_out, self.mlp_config["FULL_GRID_MEMCFG"])
                w3_out = ttnn.to_memory_config(w3_out, self.mlp_config["FULL_GRID_MEMCFG"])
            else:
                w1_out = ttnn.reshape(w1_out, (1, 1, seq_len, -1))

                w1_out = tt_all_reduce(
                    w1_out,
                    self.mesh_device,
                    cluster_axis=1,
                    num_links=2,
                )

                w3_out = ttnn.reshape(w3_out, (1, 1, seq_len, -1))
                w3_out = tt_all_reduce(
                    w3_out,
                    self.mesh_device,
                    cluster_axis=1,
                    num_links=2,
                )

        w2_in = ttnn.multiply(
            w1_out,
            w3_out,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
            input_tensor_a_activation=ttnn.UnaryOpType.SILU,
            dtype=ttnn.bfloat8_b,
        )
        if (
            mode == "decode"
        ):  # TODO Add a check for a match between FF1/FF3 and FF2 memory configs. If they match avoid doing the reshard
            # Reshard w2_in to a different core_grid configuration. Avoid using ttnn.reshard() due to incompatibility with trace mode
            if TG:
                w2_in = ttnn.to_memory_config(w2_in, self.mlp_config["FF2_ACT_MEMCFG"])
            else:
                w2_in = ttnn.sharded_to_interleaved(w2_in, ttnn.L1_MEMORY_CONFIG)
                w2_in = ttnn.interleaved_to_sharded(w2_in, self.model_config["SHARDED_MLP2_INPUT_MEMCFG"])

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        if TG and mode == "prefill":
            w2_in = ttnn.reshape(w2_in, (1, batch_dim, seq_len // batch_dim, -1))

        # This uses HiFi2 for full precision as it is dram-bound and uses bfp8 inputs
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=self.args.compute_kernel_config_hifi2,
            core_grid=ttnn.CoreGrid(y=1, x=8) if not pc_2 else None,
            dtype=ttnn.bfloat16,
            program_config=pc_2,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(w2_in)

        if mode == "decode":
            if TG:
                w2_out = tt_sharded_all_reduce(
                    w2_out,
                    self.mesh_device,
                    cluster_axis=0,
                    num_links=2,
                    memory_config=self.mlp_config["FF2_OUT_GATHERED_MEMCFG"],
                )

                w2_out = ttnn.to_memory_config(w2_out, self.mlp_config["FF1_ACT_MEMCFG"])
                return w2_out
            else:
                w2_out = ttnn.sharded_to_interleaved(
                    w2_out, ttnn.L1_MEMORY_CONFIG
                )  # FIXME: When h is L1 interleaved in decoder, this call corrupts it!

        if seq_len >= 1024:  # Reshape back to intended shape
            w2_out = ttnn.reshape(w2_out, [1, 1, seq_len, -1])

        # All reduce
        if self.args.is_multichip:
            if TG:
                w2_out_reduced = tt_all_reduce(
                    w2_out,
                    self.mesh_device,
                    cluster_axis=0,
                    num_links=2,
                )
            else:
                w2_out_reduced = ttnn.reduce_scatter(
                    w2_out,
                    scatter_dim=3,
                    math_op=ttnn.ReduceType.Sum,
                    num_links=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG if mode == "prefill" else ttnn.L1_MEMORY_CONFIG,
                )
            ttnn.deallocate(w2_out)
            return w2_out_reduced
        else:
            return w2_out
