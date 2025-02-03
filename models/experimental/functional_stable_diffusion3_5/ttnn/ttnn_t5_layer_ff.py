# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from torch import nn
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_t5_dense_gated_act_dense import (
    ttnn_T5DenseGatedActDense,
)
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_t5_layer_norm import ttnn_T5LayerNorm


class ttnn_T5LayerFF:
    def __init__(self, config, parameters=None):
        if config.is_gated_act:
            self.DenseReluDense = ttnn_T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.layer_norm = ttnn_T5LayerNorm(parameters.layer_norm, eps=config.layer_norm_epsilon)

    def __call__(self, hidden_states, parameters):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states, parameters=parameters["DenseReluDense"])
        hidden_states = hidden_states + forwarded_states
        ttnn.deallocate(forwarded_states)
        return hidden_states
