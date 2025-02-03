# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


ACT2CLS = {
    # "gelu": GELUActivation,
    # "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    # "gelu_fast": FastGELUActivation,
    "gelu_new": ttnn.gelu,
    # "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    # "gelu_pytorch_tanh": PytorchGELUTanh,
    # "gelu_accurate": AccurateGELUActivation,
    # "laplace": LaplaceActivation,
    # "leaky_relu": nn.LeakyReLU,
    # "linear": LinearActivation,
    # "mish": MishActivation,
    # "quick_gelu": QuickGELUActivation,
    # "relu": nn.ReLU,
    # "relu2": ReLUSquaredActivation,
    # "relu6": nn.ReLU6,
    # "sigmoid": nn.Sigmoid,
    # "silu": nn.SiLU,
    # "swish": nn.SiLU,
    # "tanh": nn.Tanh,
}


class ttnn_T5DenseGatedActDense:
    def __init__(self, config):
        self.wi_0 = ttnn.linear
        self.wi_1 = ttnn.linear
        self.wo = ttnn.linear
        self.act = ACT2CLS[config.dense_act_fn]

    def __call__(self, hidden_states, parameters=None):
        hidden_gelu = self.act(
            self.wi_0(hidden_states, parameters["wi_0"]["weight"], memory_config=ttnn.L1_MEMORY_CONFIG)
        )
        hidden_linear = self.wi_1(hidden_states, parameters["wi_1"]["weight"], memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = ttnn.mul(hidden_gelu, hidden_linear, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(hidden_linear)
        ttnn.deallocate(hidden_gelu)
        hidden_states = self.wo(hidden_states, parameters["wo"]["weight"], memory_config=ttnn.L1_MEMORY_CONFIG)
        return hidden_states
