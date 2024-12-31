# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    data_gen_with_range_batch_norm,
    compare_pcc,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        # (torch.Size([1, 3, 32, 32])),
        # (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 2, 32, 32])),
    ),
)
@pytest.mark.parametrize("training", [False])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("bias", [True, False])
# @pytest.mark.parametrize("eps", [1.0, 0.0, 2.34, 1e-05])
@pytest.mark.parametrize("eps", [1.0])
def test_batch_norm(input_shapes, training, weight, bias, eps, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 5, 10, device, False)
    if not training:
        mean_data, mean_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device, False)
        var_data, var_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 20, device, False)
    else:
        mean_data = None
        mean_tensor = None
        var_data = None
        var_tensor = None
    if weight:
        weight_data, weight_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device, False)
        print("Weight has value")
    else:
        weight_data = None
        weight_tensor = None
        print("Weight has NO value")
    if bias:
        bias_data, bias_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device, False)
        print("Bias has value")
    else:
        bias_data = None
        bias_tensor = None
        print("Bias has NO value")

    tt_output_tensor_on_device = ttnn.batch_norm(
        input_tensor,
        running_mean=mean_tensor,
        running_var=var_tensor,
        training=training,
        eps=eps,
        weight=weight_tensor,
        bias=bias_tensor,
    )
    print(tt_output_tensor_on_device.shape)
    output = ttnn.to_torch(tt_output_tensor_on_device)
    print("TT to torch GROUP NORM MEAN OUTPUT : ", output, output.shape)
    # print(in_data + in_data.mean(dim=(0, 2, 3), keepdim=True)) #step 1
    # print(in_data - mean_data) #step 1
    # print(in_data.var(dim=(0, 2, 3), keepdim=True) + 0.0) #step 2
    # print(var_data + 2.34) #step 2
    # print(torch.rsqrt(var_data + 2.34)) #step 3
    # print(torch.rsqrt(in_data.var(dim=(0, 2, 3), keepdim=True) + 2.34))  # step 3
    # print((in_data - mean_data) * torch.rsqrt(var_data + 2.34))  # step 4
    torch.set_printoptions(precision=5, sci_mode=False)
    torch_result = torch.nn.functional.batch_norm(
        input=in_data,
        running_mean=mean_data,
        running_var=var_data,
        weight=weight_data,
        bias=bias_data,
        training=training,
        # momentum=momentum,
        eps=eps,
    )
    print(torch_result)
    comp_pass = compare_pcc([tt_output_tensor_on_device], [torch_result * -1])
    assert comp_pass
    # return True
