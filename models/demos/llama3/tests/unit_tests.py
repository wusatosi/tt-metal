# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn as nn

import ttnn
from models.demos.llama3.tt.model_config import TtModelArgs

# TTNN Modules
import models.demos.llama3.tt.llama_common as llama_common
from models.demos.llama3.tt.llama_mlp import TtLlamaMLP
from models.demos.llama3.tt.llama_attention import TtLlamaAttention

# Reference Modules
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import FeedForward
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import Attention

# Utils
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from loguru import logger
import os


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "input_shape, output_shape, module_class, ref_module_class, max_seq_len, batch_size",
    [
        ((1, 1, 32, 2048), (1, 1, 32, 2048), TtLlamaMLP, FeedForward, 128, 1),
        # ((1, 1, 32, 2048), (1, 1, 32, 2048), TtLlamaAttention, Attention, 128, 1),  # TODO
    ],
)
def test_llama_unit(
    input_shape, output_shape, module_class, ref_module_class, max_seq_len, batch_size, mesh_device, reset_seeds
):
    """
    Llama3 unit tests.

    Args:
        input_shape (tuple): Shape of the input tensor (e.g., (batch_size, channels, height, width)).
        output_shape (tuple): Expected shape of the output tensor.
        module_class (torch.nn.Module): The module class to evaluate.
    """

    mesh_device.enable_program_cache()
    mesh_device.enable_async(True)

    # parametrize
    dtype = ttnn.bfloat8_b
    mode = "decode"

    ### INIT MODEL ARGS

    # Load the model args
    model_args = TtModelArgs(
        mesh_device,
        dummy_weights=True,
        # optimizations=optimizations,  # TODO Add this
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
    )
    model_args.n_layers = 1  # TODO Parametrize
    state_dict = model_args.load_state_dict()  # Dummy weights being used
    ########

    # Create a random input tensor in torch and ttnn
    input_ref_tensor, input_tt_tensor = create_inputs(*input_shape)

    #### RUN REF MODEL IF NO REFERENCE OUTPUT FOUND
    # TODO parametrize
    reference_output_tensor = "models/demos/llama3/tests/reference_outputs/mlp.pt"
    if os.path.exists(reference_output_tensor):
        output_ref = torch.load(reference_output_tensor)
        logger.info(f"Reference output tensor found at {reference_output_tensor}")
    else:  # Only run the reference model if no reference output is found
        logger.info(f"Reference output tensor not found. Running the reference model...")
        # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
        # TODO Find a way to parametrize this!
        first_layer_prefix = model_args.get_state_dict_prefix("TtLlamaMLP", 0)
        partial_state_dict = {
            k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
        }

        module_ref = ref_module_class(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        module_ref.load_state_dict(partial_state_dict)
        output_ref = module_ref(input_ref_tensor)
        torch.save(
            output_ref, "models/demos/llama3/tests/reference_outputs/mlp.pt"
        )  # Save the reference output for future runs
        # TODO encode in the tensor name the llama model, seqlen, batch, etc.
        # TODO See how test accuracy saves the tensors, since we need to reduce their size.

    ########

    #### RUN TTNN MODEL
    logger.info(f"Running TTNN model...")
    # TODO Parametrize this!
    module_ttnn = module_class(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
    )

    # Forward pass through the module
    output_tt = module_ttnn(input_tt_tensor, mode)  # TODO Make a default mode

    output_tt = ttnn.to_torch(
        output_tt,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    output_tt = output_tt[:, :1, :, :]

    passing, pcc_message = comp_pcc(output_ref, output_tt, 0.99)
    logger.info(comp_allclose(output_ref, output_tt))
    logger.info(f"PCC: {pcc_message}")

    # Check if the output tensor shape matches the expected shape
    assert (
        output_tt.shape == output_shape
    ), f"Output shape {output_tt.shape} does not match expected shape {output_shape}"

    assert passing, f"Llama_MLP output does not meet PCC requirement {0.99}: {pcc_message}."


# TODO figure out a way to standardize this between all unit tests
def create_inputs(input_shape, model_args, mesh_device, dtype, mode):
    input_ref_tensor = torch.randn(*input_shape)
    input_tt_tensor = ttnn.from_torch(
        input_ref_tensor,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(None, 3) if model_args.is_galaxy else (None, None), mesh_shape=model_args.cluster_shape
        ),  # When both dims are None, the mapper used is `ReplicateTensorToMesh`
        dtype=dtype,
        memory_config=(
            tt_model.model_config["MLP_ACT_MEMCFG"]
            if model_args.is_galaxy
            else model_args.model_config["SHARDED_MLP_INPUT_MEMCFG"]
        )
        if mode == "decode"
        else ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    return input_ref_tensor, input_tt_tensor
