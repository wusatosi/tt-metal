# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.sentence_bert.ttnn.common import ff1_matmul_program_config


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


class TtnnSentenceBertIntermediate:
    def __init__(self, parameters):
        self.dense = ttnn.linear
        self.parameters = parameters

    def __call__(self, hidden_states: ttnn.Tensor):
        p(hidden_states, "input to intermediate")
        out_intermediate = self.dense(
            hidden_states,
            self.parameters.dense.weight,
            bias=self.parameters.dense.bias,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=ff1_matmul_program_config,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                packer_l1_acc=False,
            ),
        )

        p(out_intermediate, "output to intermediate")
        return out_intermediate
