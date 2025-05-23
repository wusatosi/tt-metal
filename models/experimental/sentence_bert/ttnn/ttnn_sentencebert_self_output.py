# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.sentence_bert.ttnn.common import (
    layernorm_program_config,
    self_out_program_config,
)


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


class TtnnSentenceBertSelfOutput:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.config = config
        self.dense = ttnn.linear
        self.LayerNorm = ttnn.layer_norm

    def __call__(self, hidden_states: ttnn.Tensor, input_tensor: ttnn.Tensor):
        p(hidden_states, "in1")
        p(input_tensor, "in2")
        output = self.dense(
            hidden_states,
            self.parameters.dense.weight,
            bias=self.parameters.dense.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=self_out_program_config,
            dtype=ttnn.bfloat8_b,
        )
        p(output, "after 1st")
        # output = ttnn.reshard(output, input_tensor.memory_config())
        input_tensor = ttnn.reshard(input_tensor, output.memory_config())
        output = self.LayerNorm(
            output,
            residual_input_tensor=input_tensor,
            weight=self.parameters.LayerNorm.weight,
            bias=self.parameters.LayerNorm.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            epsilon=self.config.layer_norm_eps,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
            program_config=layernorm_program_config,
        )

        p(output, "self output is ")
        return output
