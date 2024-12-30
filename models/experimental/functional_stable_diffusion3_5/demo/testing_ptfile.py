import torch

from tests.ttnn.utils_for_testing import comp_pcc


def test_pt():
    for i in range(24):
        print("--------layer--------", i)
        torch_hidden_states = torch.load(
            "/home/ubuntu/punith_new/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/inputs_pt/hidden_states_joint_transformer_input_layer_"
            + str(i)
            + ".pt",
            map_location=torch.device("cpu"),
        )
        torch_encoder_states = torch.load(
            "/home/ubuntu/punith_new/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/inputs_pt/encoder_hidden_states_joint_transformer_input_layer_"
            + str(i)
            + ".pt",
            map_location=torch.device("cpu"),
        )
        torch_temb = torch.load(
            "/home/ubuntu/punith_new/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/inputs_pt/temb_joint_transformer_input_layer_"
            + str(i)
            + ".pt",
            map_location=torch.device("cpu"),
        )

        ttnn_hidden_states = torch.load(
            "/home/ubuntu/punith_new/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/ttnn_inputs_pt/hidden_states_inputs_layer"
            + str(i)
            + ".pt",
            map_location=torch.device("cpu"),
        )
        ttnn_encoder_states = torch.load(
            "/home/ubuntu/punith_new/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/ttnn_inputs_pt/encoder_hidden_states_inputs_layer"
            + str(i)
            + ".pt",
            map_location=torch.device("cpu"),
        )
        ttnn_temb = torch.load(
            "/home/ubuntu/punith_new/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/ttnn_inputs_pt/temb_inputs_layer"
            + str(i)
            + ".pt",
            map_location=torch.device("cpu"),
        )

        print("hidden_states", comp_pcc(torch_hidden_states, ttnn_hidden_states, pcc=0.99)[1])
        print("encoder_states", comp_pcc(torch_encoder_states, ttnn_encoder_states, pcc=0.99)[1])
        print("temb", comp_pcc(torch_temb, ttnn_temb, pcc=0.99)[1])
        print("------------------------------------")
