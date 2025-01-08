# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


ttnn.attach_golden_function(ttnn.experimental.add, golden_function=lambda a, b: a + b)
ttnn.attach_golden_function(ttnn.experimental.sub, golden_function=lambda a, b: a - b)
ttnn.attach_golden_function(ttnn.experimental.rsub, golden_function=lambda a, b: b - a)
ttnn.attach_golden_function(ttnn.experimental.pow, golden_function=lambda a, b: torch.pow(a, b))
ttnn.attach_golden_function(ttnn.experimental.mul, golden_function=lambda a, b: a * b)
ttnn.attach_golden_function(ttnn.experimental.div, golden_function=lambda a, b: torch.divide(a, b))
ttnn.attach_golden_function(ttnn.experimental.eq, golden_function=lambda a, b: torch.eq(a, b))
ttnn.attach_golden_function(ttnn.experimental.ne, golden_function=lambda a, b: torch.ne(a, b))
ttnn.attach_golden_function(ttnn.experimental.gt, golden_function=lambda a, b: torch.gt(a, b))
ttnn.attach_golden_function(ttnn.experimental.lt, golden_function=lambda a, b: torch.lt(a, b))
ttnn.attach_golden_function(ttnn.experimental.gte, golden_function=lambda a, b: torch.ge(a, b))
ttnn.attach_golden_function(ttnn.experimental.lte, golden_function=lambda a, b: torch.le(a, b))
ttnn.attach_golden_function(ttnn.experimental.ldexp, golden_function=lambda a, b: torch.ldexp(a, b))
ttnn.attach_golden_function(ttnn.experimental.logaddexp, golden_function=lambda a, b: torch.logaddexp(a, b))
ttnn.attach_golden_function(ttnn.experimental.logaddexp2, golden_function=lambda a, b: torch.logaddexp2(a, b))
ttnn.attach_golden_function(ttnn.experimental.logical_and, golden_function=lambda a, b: torch.logical_and(a, b))
ttnn.attach_golden_function(ttnn.experimental.logical_or, golden_function=lambda a, b: torch.logical_or(a, b))
ttnn.attach_golden_function(ttnn.experimental.logical_xor, golden_function=lambda a, b: torch.logical_xor(a, b))
ttnn.attach_golden_function(
    ttnn.experimental.squared_difference, golden_function=lambda a, b: torch.square(torch.sub(a, b))
)
ttnn.attach_golden_function(
    ttnn.experimental.bias_gelu, golden_function=lambda a, b: torch.nn.functional.gelu(torch.add(a, b))
)


def _golden_function_bitwise_left_shift(input_tensor_a, shift_amt, *args, **kwargs):
    import torch

    return torch.bitwise_left_shift(input_tensor_a, shift_amt)


ttnn.attach_golden_function(ttnn.experimental.bitwise_left_shift, golden_function=_golden_function_bitwise_left_shift)


def _golden_function_bitwise_right_shift(input_tensor_a, shift_amt, *args, **kwargs):
    import torch

    return torch.bitwise_right_shift(input_tensor_a, shift_amt)


ttnn.attach_golden_function(ttnn.experimental.bitwise_right_shift, golden_function=_golden_function_bitwise_right_shift)


def _golden_function_bitwise_and(input_tensor_a, value, *args, **kwargs):
    import torch

    return torch.bitwise_and(input_tensor_a, value)


ttnn.attach_golden_function(ttnn.experimental.bitwise_and, golden_function=_golden_function_bitwise_and)


def _golden_function_bitwise_or(input_tensor_a, value, *args, **kwargs):
    import torch

    return torch.bitwise_or(input_tensor_a, value)


ttnn.attach_golden_function(ttnn.experimental.bitwise_or, golden_function=_golden_function_bitwise_or)


def _golden_function_bitwise_xor(input_tensor_a, value, *args, **kwargs):
    import torch

    return torch.bitwise_xor(input_tensor_a, value)


ttnn.attach_golden_function(ttnn.experimental.bitwise_xor, golden_function=_golden_function_bitwise_xor)
