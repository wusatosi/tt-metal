# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params
from loguru import logger

# self.bias_qkv = ttnn.as_tensor(
#     torch.concat(
#         [
#             torch.chunk(self.state_dict[bias_q_str], self.num_devices)[0],
#             torch.chunk(self.state_dict[bias_k_str], self.num_devices)[0],
#             torch.chunk(self.state_dict[bias_v_str], self.num_devices)[0],
#         ],
#         dim=-1,
#     ).unsqueeze(0),
#     device=self.mesh_device,
#     mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
#     dtype=self.dtype,
#     memory_config=self.model_config["ATTN_BIAS_WEIGHTS_MEMCFG"],
#     layout=self.model_config["ATTN_B_LAYOUT_TILE"],
#     cache_file_name=cache_name("bias_qkv"),
# )


class TtAttention(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        query_dim: int,
        heads: int = 8,
        out_dim: int = None,
        kv_heads=None,
        dim_head: int = 64,
        weights_dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.head_dim = dim_head

        # print("Num heads is: ", self.heads)
        # print("dim head is: ", dim_head)

        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        q_weights = state_dict[f"{module_path}.to_q.weight"].unsqueeze(0).unsqueeze(0)
        k_weights = state_dict[f"{module_path}.to_k.weight"].unsqueeze(0).unsqueeze(0)
        v_weights = state_dict[f"{module_path}.to_v.weight"].unsqueeze(0).unsqueeze(0)

        out_weights = state_dict[f"{module_path}.to_out.0.weight"].unsqueeze(0).unsqueeze(0)
        out_bias = state_dict[f"{module_path}.to_out.0.bias"]

        # self.wqkv = ttnn.as_tensor(
        #     torch.concat(
        #         [
        #             torch.concat(
        #                 [
        #                     torch.transpose(
        #                         torch.chunk(self.state_dict[wq_str], configuration.num_devices)[i],
        #                         -2,
        #                         -1,
        #                     ),
        #                     torch.transpose(
        #                         torch.chunk(self.state_dict[wk_str], configuration.num_devices)[i],
        #                         -2,
        #                         -1,
        #                     ),
        #                     torch.transpose(
        #                         torch.chunk(self.state_dict[wv_str], configuration.num_devices)[i],
        #                         -2,
        #                         -1,
        #                     ),
        #                 ],
        #                 dim=-1,
        #             )
        #             for i in range(configuration.num_devices)
        #         ],
        #         dim=-1,
        #     ),
        #     device=self.mesh_device,
        #     mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
        #     dtype=self.dtype,
        #     memory_config=wqkv_mem_config,
        #     layout=self.model_config["ATTN_W_LAYOUT_TILE"],
        #     cache_file_name=cache_name("wqkv_sharded"),
        # )

        # logger.info(f"Creating QKV weights: Q={q_weights.shape} K={k_weights.shape} V={v_weights.shape} out={out_weights.shape}")
        self.self_attention = q_weights.shape[-1] == k_weights.shape[-1] and q_weights.shape[-1] == v_weights.shape[-1]
        if self.self_attention == True:
            fused_qkv_weights = torch.cat(
                [
                    torch.transpose(q_weights, -2, -1),
                    torch.transpose(k_weights, -2, -1),
                    torch.transpose(v_weights, -2, -1),
                ],
                dim=-1,
            )
            self.tt_qkv_weights = ttnn.from_torch(
                fused_qkv_weights, weights_dtype, device=device, layout=ttnn.TILE_LAYOUT
            )
            # print(f"QKV weights shape: {self.tt_qkv_weights.shape}")
        else:
            fused_kv_weights = torch.cat(
                [
                    torch.transpose(k_weights, -2, -1),
                    torch.transpose(v_weights, -2, -1),
                ],
                dim=-1,
            )
            self.tt_kv_weights = ttnn.from_torch(
                fused_kv_weights, weights_dtype, device=device, layout=ttnn.TILE_LAYOUT
            )
            self.tt_q_weights, _ = prepare_linear_params(device, q_weights, None, weights_dtype)
            self.tt_k_weights, _ = prepare_linear_params(device, k_weights, None, weights_dtype)
            self.tt_v_weights, _ = prepare_linear_params(device, v_weights, None, weights_dtype)

        self.tt_out_weights, self.tt_out_bias = prepare_linear_params(device, out_weights, out_bias, weights_dtype)

    def forward(self, hidden_states, attention_mask, encoder_hidden_states=None):
        if encoder_hidden_states is None:
            print("Encoder hidden states is None")
            encoder_hidden_states = hidden_states
        B = list(hidden_states.shape)[0]

        if self.self_attention:
            qkv_fused = ttnn.linear(
                hidden_states,
                self.tt_qkv_weights,
                bias=None,
            )

            (
                q_heads,
                k_heads,
                v_heads,
            ) = ttnn.experimental.nlp_create_qkv_heads(qkv_fused, num_heads=self.heads, transpose_k_heads=False)
        else:
            q_heads = ttnn.linear(
                hidden_states,
                self.tt_q_weights,
                bias=None,
            )

            kv_heads = ttnn.linear(
                encoder_hidden_states,
                self.tt_kv_weights,
                bias=None,
            )

            # std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> CreateQKVHeadsOperation::invoke(
            #     const Tensor& input_tensor,
            #     const uint32_t num_q_heads,
            #     const std::optional<uint32_t> num_kv_heads,
            #     const bool transpose_k_heads,
            #     const std::optional<MemoryConfig>& memory_config,
            #     std::optional<std::array<Tensor, 3>> optional_output_tensors) {
            #     return invoke(
            #         ttnn::DefaultQueueId,
            #         input_tensor,
            #         num_q_heads,
            #         num_kv_heads,
            #         transpose_k_heads,
            #         memory_config,
            #         std::move(optional_output_tensors));

            q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
                input=q_heads,
                input_kv=kv_heads,
                num_heads=self.heads,
                num_kv_heads=self.heads,
                transpose_k_heads=False,
            )

            # core_grid = ttnn.CoreGrid(y=8, x=8)
            # num_cores = core_grid.x * core_grid.y

            # print("Q shape: ", q_heads.shape)
            # print("KV shape: ", kv_heads.shape)
            # height_shard_q = (q_heads.shape[0] * q_heads.shape[1] * q_heads.shape[2]) // num_cores
            # height_shard_kv = 32
            # print("Height shard q: ", height_shard_q)
            # print("Height shard kv: ", height_shard_kv)
            # grid_size = core_grid
            # grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
            # shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
            # shard_spec_q = ttnn.ShardSpec(shard_grid, (height_shard_q, q_heads.shape[-1]), ttnn.ShardOrientation.ROW_MAJOR)
            # shard_spec_kv = ttnn.ShardSpec(shard_grid, (height_shard_kv, kv_heads.shape[-1]), ttnn.ShardOrientation.ROW_MAJOR)
            # mem_cfg_q = ttnn.MemoryConfig(
            #     ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec_q
            # )
            # mem_cfg_v = ttnn.MemoryConfig(
            #     ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec_kv
            # )

            # q_heads = ttnn.to_memory_config(q_heads, mem_cfg_q)
            # kv_heads = ttnn.to_memory_config(kv_heads, mem_cfg_v)

            # q_heads2, k_heads, v_heads =  ttnn.experimental.create_qkv_heads_from_separate_tensors(
            #     q_heads,
            #     kv_heads,
            #     num_heads=self.heads,
            #     num_kv_heads=self.heads,
            #     transpose_k_heads=False,
            #     memory_config = ttnn.DRAM_MEMORY_CONFIG,
            # )

            # q_heads = ttnn.linear(
            #     hidden_states,
            #     self.tt_q_weights,
            #     bias=None,
            # )
            # k_heads = ttnn.linear(
            #     encoder_hidden_states,
            #     self.tt_k_weights,
            #     bias=None,
            # )
            # v_heads = ttnn.linear(
            #     encoder_hidden_states,
            #     self.tt_v_weights,
            #     bias=None,
            # )
            # inner_dim = list(k_heads.shape)[-1]
            # head_dim = inner_dim // self.heads

            # q_heads = ttnn.reshape(q_heads, [B, -1, self.heads, head_dim])
            # q_heads = ttnn.transpose(q_heads, 1, 2)

            # k_heads = ttnn.reshape(k_heads, [B, -1, self.heads, head_dim])
            # k_heads = ttnn.transpose(k_heads, 1, 2)

            # v_heads = ttnn.reshape(v_heads, [B, -1, self.heads, head_dim])
            # v_heads = ttnn.transpose(v_heads, 1, 2)
        # query = ttnn.linear(
        #     hidden_states,
        #     self.tt_q_weights,
        #     bias=None,
        # )
        # key = ttnn.linear(
        #     encoder_hidden_states,
        #     self.tt_k_weights,
        #     bias=None,
        # )
        # value = ttnn.linear(
        #     encoder_hidden_states,
        #     self.tt_v_weights,
        #     bias=None,
        # )
        #        logger.info(f"Attention call! hidden_states shape: {hidden_states.shape} encoder_hidden_states shape: {encoder_hidden_states.shape} query shape: {query.shape} key shape: {key.shape} value shape: {value.shape}")

        # inner_dim = list(key.shape)[-1]
        # head_dim = inner_dim // self.heads

        # logger.info(f"Query shape pre reshape: {query.shape}")
        # query = ttnn.reshape(query, [B, -1, self.heads, head_dim])
        # logger.info(f"Query shape post reshape: {query.shape}")
        # query = ttnn.transpose(query, 1, 2)
        # logger.info(f"Query shape post transpose: {query.shape}")

        # key = ttnn.reshape(key, [B, -1, self.heads, head_dim])
        # key = ttnn.transpose(key, 1, 2)

        # value = ttnn.reshape(value, [B, -1, self.heads, head_dim])
        # value = ttnn.transpose(value, 1, 2)

        # logger.info(f"Q: {query.shape} K: {key.shape} V: {value.shape} pre sdpa")
        # logger.info(f"q_heads: {q_heads.shape} k_heads: {k_heads.shape} v_heads: {v_heads.shape}")
        hidden_states = ttnn.transformer.scaled_dot_product_attention(
            # query,
            # key,
            # value,
            q_heads,
            k_heads,
            v_heads,
            is_causal=False,
            attn_mask=attention_mask,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        hidden_states = ttnn.transpose(hidden_states, 1, 2)
        hidden_states = ttnn.reshape(hidden_states, [B, -1, self.heads * self.head_dim])

        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_out_weights,
            bias=self.tt_out_bias,
        )
        logger.info(f"Attention call! out shape: {hidden_states.shape}")

        return hidden_states
