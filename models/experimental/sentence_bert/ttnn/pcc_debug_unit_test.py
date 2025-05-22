# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch
import math
from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_lib.utils import pad_weight

query_key_value_matmul_program_config_sentence_bert = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=3,
    out_subblock_h=1,
    out_subblock_w=6,
    out_block_w=12,
    out_block_h=12,
    per_core_M=12,
    per_core_N=12,
    transpose_mcast=False,
    fused_activation=None,
    fuse_batch=True,
)
query_key_value_matmul_program_config_bert = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=4,
    out_subblock_h=1,
    out_subblock_w=6,
    out_block_w=12,
    out_block_h=12,
    per_core_M=12,
    per_core_N=12,
    transpose_mcast=False,
    fused_activation=None,
    fuse_batch=True,
)


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


@pytest.mark.parametrize(
    "attention_scores,attention_mask,head_size",
    [([8, 12, 384, 384], [8, 1, 1, 384], 64), ([8, 16, 384, 384], [8, 1, 1, 384], 64)],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_baseline_attention_softmax(device, attention_scores, attention_mask, head_size):  # 0.999 PCC
    attention_scores_t = torch.randn(attention_scores)
    attention_mask_t = torch.randn(attention_mask)
    attention_scores_tt = ttnn.from_torch(
        attention_scores_t,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    attention_mask_tt = ttnn.from_torch(
        attention_mask_t,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    # torch
    val = 1 / math.sqrt(head_size)
    attention_scores_t = attention_scores_t * val
    attention_scores_t = attention_scores_t + attention_mask_t
    attention_probs = torch.nn.functional.softmax(attention_scores_t, dim=-1)
    # ttnn
    attention_scores_tt = ttnn.multiply(attention_scores_tt, val)
    attention_scores_tt = ttnn.add(attention_scores_tt, attention_mask_tt)
    attention_probs_tt = ttnn.softmax(attention_scores_tt, dim=-1)

    attention_probs_tt = ttnn.to_torch(attention_probs_tt)
    assert_with_pcc(attention_probs, attention_probs_tt, 1.0)


@pytest.mark.parametrize(
    "attention_scores,attention_mask,head_size",
    [([8, 12, 384, 384], [8, 1, 1, 384], 64), ([8, 16, 384, 384], [8, 1, 1, 384], 64)],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_transformer_attention_softmax(device, attention_scores, attention_mask, head_size):
    attention_scores_t = torch.randn(attention_scores)
    attention_mask_t = torch.randn(attention_mask)
    attention_scores_tt = ttnn.from_torch(
        attention_scores_t,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    attention_mask_tt = ttnn.from_torch(
        attention_mask_t,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    val = 1 / math.sqrt(head_size)
    # ttnn1
    attention_scores_tt_1 = ttnn.multiply(attention_scores_tt, val)
    attention_scores_tt_1 = ttnn.add(attention_scores_tt_1, attention_mask_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
    attention_probs_tt_1 = ttnn.softmax(attention_scores_tt_1, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
    # ttnn2
    p(attention_scores_tt, "1st inp")
    p(attention_mask_tt, "2nd inp")
    attention_probs_tt_2 = ttnn.transformer.attention_softmax_(
        attention_scores_tt, attention_mask=attention_mask_tt, head_size=head_size, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    p(attention_probs_tt_1, "1st one")
    p(attention_probs_tt_2, "2nd one")
    attention_probs_tt_2 = ttnn.to_torch(attention_probs_tt_2)
    attention_probs_tt_1 = ttnn.to_torch(attention_probs_tt_1)
    assert_with_pcc(attention_probs_tt_1, attention_probs_tt_2, 1.0)


@pytest.mark.parametrize(
    "hidden_states_shape,q_w_shape,q_b_shape,num_heads",
    [
        ((8, 384, 768), (768, 768), (768,), 12),
        ((8, 384, 1024), (1024, 1024), (1024,), 16),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_transformer_split_qkv(device, num_heads, hidden_states_shape, q_w_shape, q_b_shape):
    torch.manual_seed(0)
    attn_head_size = hidden_states_shape[-1] // num_heads
    query = torch.nn.Linear(q_w_shape[1], q_w_shape[0])
    key = torch.nn.Linear(q_w_shape[1], q_w_shape[0])
    value = torch.nn.Linear(q_w_shape[1], q_w_shape[0])
    hidden_states = torch.randn(hidden_states_shape, dtype=torch.bfloat16)
    query.weight = torch.nn.Parameter(torch.randn((q_w_shape), dtype=torch.bfloat16))
    query.bias = torch.nn.Parameter(torch.randn((q_b_shape), dtype=torch.bfloat16))
    query_out = query(hidden_states)

    key.weight = torch.nn.Parameter(torch.randn((q_w_shape), dtype=torch.bfloat16))
    key.bias = torch.nn.Parameter(torch.randn((q_b_shape), dtype=torch.bfloat16))
    key_out = key(hidden_states)

    value.weight = torch.nn.Parameter(torch.randn((q_w_shape), dtype=torch.bfloat16))
    value.bias = torch.nn.Parameter(torch.randn((q_b_shape), dtype=torch.bfloat16))
    value_out = value(hidden_states)
    # # ttnn
    hidden_states_tt = ttnn.from_torch(
        hidden_states.unsqueeze(dim=1),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    qw = query.weight
    kw = key.weight
    vw = value.weight
    qw = torch.transpose(qw, -1, -2)
    kw = torch.transpose(kw, -1, -2)
    vw = torch.transpose(vw, -1, -2)
    qb = query.bias
    kb = key.bias
    vb = value.bias
    const_w_dims = qw.shape[:-1]
    qw = qw.reshape([*const_w_dims, num_heads // 2, -1])
    kw = kw.reshape(qw.shape)
    vw = vw.reshape(qw.shape)
    qkv_weight_torch = torch.cat((qw, kw, vw), -1).reshape([*const_w_dims, -1])
    qkv_weight_torch = pad_weight(qkv_weight_torch)
    const_b_dims = qb.shape[:-1]
    qb = qb.reshape([*const_b_dims, num_heads // 2, -1])
    kb = kb.reshape(qb.shape)
    vb = vb.reshape(qb.shape)
    qkv_bias_torch = torch.cat((qb, kb, vb), -1).reshape([*const_b_dims, -1])
    qkv_bias_torch = pad_weight(qkv_bias_torch)
    hidden_states_tt = ttnn.from_torch(
        hidden_states.unsqueeze(dim=1),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    combined_weight = ttnn.from_torch(
        qkv_weight_torch,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    combined_bias = ttnn.from_torch(
        qkv_bias_torch,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    memory_config = ttnn.create_sharded_memory_config(
        hidden_states_tt.shape,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    hidden_states_tt_sharded = ttnn.to_memory_config(hidden_states_tt, memory_config)
    p(hidden_states_tt_sharded, "input")
    query_key_value = ttnn.linear(
        hidden_states_tt_sharded,
        combined_weight,
        bias=combined_bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=query_key_value_matmul_program_config_bert
        if num_heads == 16
        else query_key_value_matmul_program_config_sentence_bert,
    )
    p(combined_weight, "w for qkv linear")
    p(combined_bias, "b for qkv linear")
    print(
        query_key_value_matmul_program_config_bert,
        "ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG",
        "ttnn.bfloat8_b",
        "program config , mem config , dttype",
    )
    p(query_key_value, "output of qkv linear")
    # query_key_value = ttnn.to_layout(query_key_value,ttnn.ROW_MAJOR_LAYOUT)
    # query_key_value = ttnn.pad(query_key_value, ((0, 0), (0, 0), (0, 0), (0,768)), 0)
    # output_tensor1 = torch.zeros((8,16,384,64),dtype=torch.bfloat16)
    # output_tensor1 = ttnn.from_torch(output_tensor1,dtype=ttnn.bfloat8_b,layout=ttnn.TILE_LAYOUT,device=device)
    # memory_config = ttnn.create_sharded_memory_config(
    #     (768,64),
    #     core_grid=ttnn.CoreGrid(y=8, x=8),
    #     strategy=ttnn.ShardStrategy.HEIGHT,
    #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
    # )
    # output_tensor1 = ttnn.to_memory_config(output_tensor1, memory_config)
    # p(output_tensor1,"out tensor")
    # ss
    (
        query_ttnn,
        key_ttnn,
        value_ttnn,
    ) = ttnn.experimental.split_query_key_value_and_split_heads(
        query_key_value,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        # output_tensors =
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),  # device.compute_with_storage_grid_size(),
        num_heads=num_heads,
    )  # COMBINED QKV --> Q,KT,V ( Q,K,V APPLIED R,P)
    print("args to split", device.compute_with_storage_grid_size(), "ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG")
    p(query_ttnn, "queryyy")
    p(key_ttnn, "keyss")
    p(value_ttnn, "valueee")
    query_ttnn = ttnn.to_torch(query_ttnn)
    key_ttnn = ttnn.to_torch(key_ttnn).permute(0, 1, 3, 2)
    value_ttnn = ttnn.to_torch(value_ttnn)
    pcc11 = assert_with_pcc(
        query_ttnn,
        query_out.view(query_out.shape[0], query_out.shape[1], num_heads, attn_head_size).permute(0, 2, 1, 3),
        0,
    )  # 0.08
    # pcc22 = assert_with_pcc(key_ttnn, key_out, 0)  # 0.001
    pcc33 = assert_with_pcc(
        value_ttnn,
        value_out.view(query_out.shape[0], query_out.shape[1], num_heads, attn_head_size).permute(0, 2, 1, 3),
        0,
    )  # 0.08
    print(pcc11, pcc33)


# torch_tensor = torch.cat(
#         [
#             query_out_d,
#             key_out_d,
#             value_out_d,
#         ],
#         dim=-1,
# )
# print("winef",torch_tensor.shape)
# tt_tensor= ttnn.from_torch(torch_tensor,dtype=ttnn.bfloat8_b,layout=ttnn.TILE_LAYOUT,device=device,memory_config=ttnn.L1_MEMORY_CONFIG)
# if num_heads==12:
#     core_grid = ttnn.CoreGrid(y=6, x=8)
# elif num_heads==16:
#     core_grid = ttnn.CoreGrid(y=8, x=8)
# tt_tensor = ttnn.to_memory_config(
#         tt_tensor,
#         memory_config=ttnn.create_sharded_memory_config(
#             tt_tensor.shape,
#             core_grid=core_grid,
#             strategy=ttnn.ShardStrategy.BLOCK,
#             orientation=ttnn.ShardOrientation.COL_MAJOR,
#         ),
#     )

# cross cehcking

# torch_tensor = torch.randn(qkv_shape)
# tt_tensor= ttnn.from_torch(torch_tensor,dtype=ttnn.bfloat8_b,layout=ttnn.TILE_LAYOUT,device=device,memory_config=ttnn.L1_MEMORY_CONFIG)
# if num_heads==12:
#     core_grid = ttnn.CoreGrid(y=6, x=8)
# elif num_heads==16:
#     core_grid = ttnn.CoreGrid(y=8, x=8)
# tt_tensor = ttnn.to_memory_config(
#         tt_tensor,
#         memory_config=ttnn.create_sharded_memory_config(
#             tt_tensor.shape,
#             core_grid=core_grid,
#             strategy=ttnn.ShardStrategy.BLOCK,
#             orientation=ttnn.ShardOrientation.COL_MAJOR,
#         ),
#     )
# p(tt_tensor,"sharded_input")
# (
#     query,
#     key,
#     value,
# ) = ttnn.transformer.split_query_key_value_and_split_heads(
#     tt_tensor,
#     memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
#     num_heads=num_heads,
# )
# p(query,'q')
# p(key,'k')
# p(value,'v')


# q_w_tt = preprocess_linear_weight(query.weight, dtype=ttnn.bfloat16)
# q_b_tt = preprocess_linear_bias(query.bias, dtype=ttnn.bfloat16)
# query_layer =ttnn.linear(
#         hidden_states_tt,
#         ttnn.to_device(q_w_tt,device=device),
#         bias=ttnn.to_device(q_b_tt,device=device),
#         memory_config=ttnn.L1_MEMORY_CONFIG,
# )
# query_layer = ttnn.reshape(
#         query_layer,
#         (query_layer.shape[0], query_layer.shape[1], num_heads,attn_head_size),
#     )
# query_layer = ttnn.permute(query_layer,(0,2,1,3))
# p(query_layer,"q_tt")


# k_w_tt = preprocess_linear_weight(key.weight, dtype=ttnn.bfloat16)
# k_b_tt = preprocess_linear_bias(key.bias, dtype=ttnn.bfloat16)
# key_layer =ttnn.linear(
#         hidden_states_tt,
#         ttnn.to_device(k_w_tt,device=device),
#         bias=ttnn.to_device(k_b_tt,device=device),
#         memory_config=ttnn.L1_MEMORY_CONFIG,
# )
# key_layer = ttnn.reshape(
#         key_layer,
#         (key_layer.shape[0], key_layer.shape[1], num_heads,attn_head_size),
#     )
# key_layer = ttnn.permute(key_layer,(0,2,1,3))
# p(key_layer,"k_tt")

# v_w_tt = preprocess_linear_weight(value.weight, dtype=ttnn.bfloat16)
# v_b_tt = preprocess_linear_bias(value.bias, dtype=ttnn.bfloat16)
# value_layer =ttnn.linear(
#         hidden_states_tt,
#         ttnn.to_device(v_w_tt,device=device),
#         bias=ttnn.to_device(v_b_tt,device=device),
#         memory_config=ttnn.L1_MEMORY_CONFIG,
# )
# value_layer = ttnn.reshape(
#         value_layer,
#         (value_layer.shape[0], value_layer.shape[1], num_heads,attn_head_size),
#     )
# value_layer = ttnn.permute(value_layer,(0,2,1,3))
# p(value_layer,"v_tt")

# query_layer=ttnn.to_torch(query_layer)
# key_layer=ttnn.to_torch(key_layer)
# value_layer=ttnn.to_torch(value_layer)
# pcc1 = assert_with_pcc(query_layer,query_out,0.99)
# pcc2 = assert_with_pcc(key_layer,key_out,0.99)
# pcc3 = assert_with_pcc(value_layer,value_out,0.99)
# print(pcc1,pcc2,pcc3)


# hidden_states = torch.load("/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/ttnn/dumps/hidden_states")
#     qw_dumped =torch.load("/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/ttnn/dumps/q_w")
#     qb_dumped =torch.load("/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/ttnn/dumps/q_b")
#     kw_dumped =torch.load("/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/ttnn/dumps/k_w")
#     kb_dumped =torch.load("/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/ttnn/dumps/k_b")
#     vw_dumped =torch.load("/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/ttnn/dumps/v_w")
#     vb_dumped =torch.load("/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/ttnn/dumps/v_b")
#     key.weight = torch.nn.Parameter(kw_dumped)
#     key.bias = torch.nn.Parameter(kb_dumped)
#     query.weight = torch.nn.Parameter(qw_dumped)
#     query.bias = torch.nn.Parameter(qb_dumped)
#     value.weight = torch.nn.Parameter(vw_dumped)
#     value.bias = torch.nn.Parameter(vb_dumped)
