# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range_batch_norm,
    compare_results_batch_norm,
)
from itertools import product
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "input_shapes",
    [
        # torch.Size([7, 8, 23, 23]),
        # torch.Size([7, 8, 1026, 1026]),
        torch.Size([7, 8, 1026 * 4, 1026 * 4]),
    ],
)
@pytest.mark.parametrize(
    "training, check_mean, check_var",
    [
        (True, True, True),
        # (False, True, True),
    ],
)
@pytest.mark.parametrize("weight", [True])
@pytest.mark.parametrize("bias", [True])
@pytest.mark.parametrize("eps", [1e-05])
@pytest.mark.parametrize("momentum", [0.1])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 1824800}], indirect=True)
def test_batch_norm_fp32(
    input_shapes, check_mean, check_var, weight, bias, eps, device, momentum, training, testing_dtype="float32"
):
    in_data, input_tensor = data_gen_with_range_batch_norm(
        input_shapes, 5, 10, device, is_input=True, testing_dtype=testing_dtype
    )
    mean_data, mean_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 10, device, testing_dtype=testing_dtype)
        if (check_mean)
        else (None, None)
    )
    var_data, var_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 20, device, testing_dtype=testing_dtype)
        if (check_var)
        else (None, None)
    )
    weight_data, weight_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 10, device, testing_dtype=testing_dtype)
        if weight
        else (None, None)
    )
    bias_data, bias_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 10, device, testing_dtype=testing_dtype)
        if bias
        else (None, None)
    )

    if (not training) and ((not check_mean) or (not check_var)):
        pytest.xfail("running_mean and running_var must be defined in evaluation mode")

    tt_output_tensor_on_device = ttnn.batch_norm(
        input_tensor,
        running_mean=mean_tensor,
        running_var=var_tensor,
        training=training,
        eps=eps,
        weight=weight_tensor,
        bias=bias_tensor,
        momentum=momentum,
    )
    tt_output = ttnn.to_torch(tt_output_tensor_on_device)
    tt_updated_mean = None
    tt_updated_var = None
    if training:
        if check_mean:
            tt_updated_mean = ttnn.to_torch(mean_tensor)
        if check_var:
            tt_updated_var = ttnn.to_torch(var_tensor)

    torch_result = torch.nn.functional.batch_norm(
        input=in_data,
        running_mean=mean_data,
        running_var=var_data,
        weight=weight_data,
        bias=bias_data,
        training=training,
        eps=eps,
        momentum=momentum,
    )
    comp_BN_Output = compare_results_batch_norm([tt_output], [torch_result])
    if training:
        channels = input_shapes[1]
        if check_mean:
            comp_BN_running_mean = compare_results_batch_norm(
                [tt_updated_mean], [mean_data.view(1, channels, 1, 1)], stats=True
            )  # Check Updated running mean
        else:
            if tt_updated_mean is None:
                comp_BN_running_mean = True
            else:
                comp_BN_running_mean = False
        if check_var:
            comp_BN_running_var = compare_results_batch_norm(
                [tt_updated_var], [var_data.view(1, channels, 1, 1)], stats=True
            )  # Check Updated running var
        else:
            if tt_updated_var is None:
                comp_BN_running_var = True
            else:
                comp_BN_running_var = False
        comp_BN_Output = comp_BN_Output and comp_BN_running_mean and comp_BN_running_var
    assert comp_BN_Output


# @pytest.mark.parametrize(
#     "input_shapes",
#     [
#         # torch.Size([7, 8, 23, 23]),
#         # torch.Size([7, 8, 1026, 1026]),
#         torch.Size([7, 8, 1026*4, 1026*4]),
#     ],
# )
# @pytest.mark.parametrize(
#     "training, check_mean, check_var",
#     [
#         (True, True, True),
#         # (False, True, True),
#     ],
# )
# @pytest.mark.parametrize("weight", [True])
# @pytest.mark.parametrize("bias", [True])
# @pytest.mark.parametrize("eps", [1e-05])
# @pytest.mark.parametrize("momentum", [0.1])
# def test_batch_norm_bf16(input_shapes, training, check_mean, check_var, weight, bias, eps, momentum, device):
#     in_data, input_tensor = data_gen_with_range_batch_norm(input_shapes, 5, 10, device, is_input=True)
#     mean_data, mean_tensor = (
#         data_gen_with_range_batch_norm(input_shapes, 4, 10, device) if (check_mean) else (None, None)
#     )
#     var_data, var_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 20, device) if (check_var) else (None, None)
#     weight_data, weight_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device) if weight else (None, None)
#     bias_data, bias_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device) if bias else (None, None)

#     if (not training) and ((not check_mean) or (not check_var)):
#         pytest.xfail("running_mean and running_var must be defined in evaluation mode")

#     tt_output_tensor_on_device = ttnn.batch_norm(
#         input_tensor,
#         running_mean=mean_tensor,
#         running_var=var_tensor,
#         training=training,
#         eps=eps,
#         momentum=momentum,
#         weight=weight_tensor,
#         bias=bias_tensor,
#     )
#     tt_output = ttnn.to_torch(tt_output_tensor_on_device)
#     tt_updated_mean = None
#     tt_updated_var = None
#     if training:
#         if check_mean:
#             tt_updated_mean = ttnn.to_torch(mean_tensor)
#         if check_var:
#             tt_updated_var = ttnn.to_torch(var_tensor)

#     torch_result = torch.nn.functional.batch_norm(
#         input=in_data,
#         running_mean=mean_data,
#         running_var=var_data,
#         weight=weight_data,
#         bias=bias_data,
#         training=training,
#         eps=eps,
#         momentum=momentum,
#     )
#     comp_BN_Output = compare_results_batch_norm([tt_output], [torch_result])  # Check BN Result
#     if training:
#         channels = input_shapes[1]
#         if check_mean:
#             comp_BN_running_mean = compare_results_batch_norm(
#                 [tt_updated_mean], [mean_data.view(1, channels, 1, 1)], stats=True
#             )  # Check Updated running mean
#         else:
#             if tt_updated_mean is None:
#                 comp_BN_running_mean = True
#             else:
#                 comp_BN_running_mean = False
#         if check_var:
#             comp_BN_running_var = compare_results_batch_norm(
#                 [tt_updated_var], [var_data.view(1, channels, 1, 1)], stats=True
#             )  # Check Updated running var
#         else:
#             if tt_updated_var is None:
#                 comp_BN_running_var = True
#             else:
#                 comp_BN_running_var = False
#         comp_BN_Output = comp_BN_Output and comp_BN_running_mean and comp_BN_running_var

#     assert comp_BN_Output
