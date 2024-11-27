# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.optim as optim

import ttnn

import pytest
from models.utility_functions import (
    comp_allclose_and_pcc,
)
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_ttnn,
)
from loguru import logger


def create_tt_tensor(tensor: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device)


def create_tt_tensors(cpu_grad, cpu_weight, cpu_exp_avg, cpu_exp_avg_sq, cpu_max_exp_avg_sq, amsgrad, dtype, device):
    def create_tt_tensor(tensor: torch.Tensor, dtype, device):
        return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # input tensors
    param_in = create_tt_tensor(cpu_weight, dtype, device)
    grad = create_tt_tensor(cpu_grad, dtype, device)
    exp_avg_in = create_tt_tensor(cpu_exp_avg, dtype, device)
    exp_avg_sq_in = create_tt_tensor(cpu_exp_avg_sq, dtype, device)
    max_exp_avg_sq_in = create_tt_tensor(cpu_max_exp_avg_sq, dtype, device) if amsgrad else None

    # output tensors
    param_out = create_tt_tensor(cpu_weight, dtype, device)
    exp_avg_out = create_tt_tensor(cpu_exp_avg, dtype, device)
    exp_avg_sq_out = create_tt_tensor(cpu_exp_avg_sq, dtype, device)
    max_exp_avg_sq_out = create_tt_tensor(cpu_max_exp_avg_sq, dtype, device) if amsgrad else None

    return (
        (param_in, grad, exp_avg_in, exp_avg_sq_in, max_exp_avg_sq_in),
        (param_out, exp_avg_out, exp_avg_sq_out, max_exp_avg_sq_out),
    )


def run_moreh_adamw(
    shape,
    lr,
    betas,
    eps,
    weight_decay,
    amsgrad,
    step,
    fp32_dest_acc_en,
    device,
    *,
    ttnn_dtype=ttnn.bfloat16,
    torch_dtype=torch.bfloat16,
):
    compute_kernel_config = get_compute_kernel_options(fp32_dest_acc_en)

    x_data = torch.rand(shape).to(torch_dtype)
    y_data = torch.rand(shape).to(torch_dtype)

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.weight = nn.Parameter(torch.randn(shape).to(torch_dtype)).to(torch_dtype)

        def forward(self, x):
            return torch.mul(x, self.weight)

    model = SimpleModel()
    criterion = nn.L1Loss()
    optimizer = optim.AdamW({model.weight}, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    # until step-1
    for _ in range(step - 1):
        optimizer.zero_grad()
        outputs = model(x_data)
        loss = criterion(outputs, y_data)
        loss.backward()
        optimizer.step()

    cpu_weight = model.weight.clone()
    if step == 1:
        cpu_exp_avg = torch.zeros_like(model.weight)
        cpu_exp_avg_sq = torch.zeros_like(model.weight)
        cpu_max_exp_avg_sq = torch.zeros_like(model.weight)
    else:
        optimizer_state_dict = optimizer.state_dict()
        cpu_exp_avg = optimizer_state_dict["state"][0]["exp_avg"].clone()
        cpu_exp_avg_sq = optimizer_state_dict["state"][0]["exp_avg_sq"].clone()
        if amsgrad:
            cpu_max_exp_avg_sq = optimizer_state_dict["state"][0]["max_exp_avg_sq"].clone()
        else:
            cpu_max_exp_avg_sq = None

    # last step
    optimizer.zero_grad()
    outputs = model(x_data)
    loss = criterion(outputs, y_data)
    loss.backward()
    optimizer.step()

    cpu_grad = model.weight.grad.clone()
    optimizer_state_dict = optimizer.state_dict()
    cpu_exp_avg_result = optimizer_state_dict["state"][0]["exp_avg"].clone()
    cpu_exp_avg_sq_result = optimizer_state_dict["state"][0]["exp_avg_sq"].clone()
    if amsgrad:
        cpu_max_exp_avg_sq_result = optimizer_state_dict["state"][0]["max_exp_avg_sq"].clone()
    else:
        cpu_max_exp_avg_sq_result = None

    tt_input_tensors, tt_output_tensors = create_tt_tensors(
        cpu_grad,
        cpu_weight,
        cpu_exp_avg,
        cpu_exp_avg_sq,
        cpu_max_exp_avg_sq,
        amsgrad,
        ttnn_dtype,
        device,
    )

    tt_param_in, tt_grad, tt_exp_avg_in, tt_exp_avg_sq_in, tt_max_exp_avg_sq_in = tt_input_tensors
    tt_param_out, tt_exp_avg_out, tt_exp_avg_sq_out, tt_max_exp_avg_sq_out = tt_output_tensors

    ret_list_ = ttnn.operations.moreh.adamw(
        tt_param_in,
        tt_grad,
        tt_exp_avg_in,
        tt_exp_avg_sq_in,
        lr,
        betas[0],
        betas[1],
        eps,
        weight_decay,
        step,
        amsgrad,
        max_exp_avg_sq_in=tt_max_exp_avg_sq_in,
        param_out=tt_param_out,
        exp_avg_out=tt_exp_avg_out,
        exp_avg_sq_out=tt_exp_avg_sq_out,
        max_exp_avg_sq_out=tt_max_exp_avg_sq_out,
        compute_kernel_config=compute_kernel_config,
    )

    param_result = ttnn.to_torch(tt_param_out).reshape(shape)
    exp_avg_result = ttnn.to_torch(tt_exp_avg_out).reshape(shape)
    exp_avg_sq_result = ttnn.to_torch(tt_exp_avg_sq_out).reshape(shape)

    if amsgrad:
        max_exp_avg_sq_result = ttnn.to_torch(tt_max_exp_avg_sq_out).reshape(shape)
    else:
        max_exp_avg_sq_result = None

    whole_passing = True

    rtol = atol = 0.1
    pcc = 0.99
    passing, out = comp_allclose_and_pcc(model.weight, param_result, pcc=pcc, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")
    whole_passing &= passing

    passing, out = comp_allclose_and_pcc(cpu_exp_avg_result, exp_avg_result, pcc=pcc, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (exp_avg)={passing}")
    logger.debug(f"Output pcc={out}")
    whole_passing &= passing

    passing, out = comp_allclose_and_pcc(cpu_exp_avg_sq_result, exp_avg_sq_result, pcc=pcc, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (exp_avg_sq)={passing}")
    logger.debug(f"Output pcc={out}")
    whole_passing &= passing

    if amsgrad:
        passing, out = comp_allclose_and_pcc(
            cpu_max_exp_avg_sq_result, max_exp_avg_sq_result, pcc=pcc, rtol=rtol, atol=atol
        )
        logger.debug(f"Out passing (max_exp_avg_sq)={passing}")
        logger.debug(f"Output pcc={out}")
        whole_passing &= passing

    assert whole_passing


def run_moreh_adam(shape, lr, betas, eps, weight_decay, amsgrad, fp32_dest_acc_en, device, dtype=ttnn.bfloat16, step=1):
    x_data = torch.rand(shape).to(torch.bfloat16)
    y_data = torch.rand(shape).to(torch.bfloat16)

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.weight = nn.Parameter(torch.randn(shape).to(torch.bfloat16)).to(torch.bfloat16)

        def forward(self, x):
            return torch.mul(x, self.weight)

    model = SimpleModel()

    cpu_exp_avg = torch.zeros_like(model.weight)
    cpu_exp_avg_sq = torch.zeros_like(model.weight)
    cpu_max_exp_avg_sq = torch.zeros_like(model.weight)

    dev_param = create_tt_tensor(model.weight, device, dtype=dtype)
    dev_exp_avg = create_tt_tensor(cpu_exp_avg, device, dtype=dtype)
    dev_exp_avg_sq = create_tt_tensor(cpu_exp_avg_sq, device, dtype=dtype)
    dev_max_exp_avg_sq = create_tt_tensor(cpu_max_exp_avg_sq, device, dtype=dtype)

    dev_param_out = create_tt_tensor(model.weight, device, dtype=dtype)
    dev_exp_avg_out = create_tt_tensor(cpu_exp_avg, device, dtype=dtype)
    dev_exp_avg_sq_out = create_tt_tensor(cpu_exp_avg_sq, device, dtype=dtype)
    dev_max_exp_avg_sq_out = create_tt_tensor(cpu_max_exp_avg_sq, device, dtype=dtype)

    criterion = nn.L1Loss()
    optimizer = optim.Adam({model.weight}, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    optimizer.zero_grad()
    optimizer_state_dict = optimizer.state_dict()
    outputs = model(x_data)
    loss = criterion(outputs, y_data)
    loss.backward()

    cpu_grad = model.weight.grad.clone()
    dev_grad = create_tt_tensor(cpu_grad, device, dtype=dtype)

    for _ in range(step):
        optimizer.step()

    optimizer_state_dict = optimizer.state_dict()

    cpu_exp_avg_result = optimizer_state_dict["state"][0]["exp_avg"]
    cpu_exp_avg_sq_result = optimizer_state_dict["state"][0]["exp_avg_sq"]
    if "max_exp_avg_sq" in optimizer_state_dict["state"][0]:
        cpu_max_exp_avg_sq_result = optimizer_state_dict["state"][0]["max_exp_avg_sq"]
    else:
        cpu_max_exp_avg_sq_result = None

    compute_kernel_config = get_compute_kernel_options(fp32_dest_acc_en)

    (
        dev_param_out,
        dev_exp_avg_out,
        dev_exp_avg_sq_out,
        dev_max_exp_avg_sq_out,
    ) = ttnn.operations.moreh.adam(
        dev_param,
        dev_grad,
        dev_exp_avg,
        dev_exp_avg_sq,
        lr=lr,
        beta1=betas[0],
        beta2=betas[1],
        eps=eps,
        weight_decay=weight_decay,
        step=step,
        amsgrad=amsgrad,
        max_exp_avg_sq_in=dev_max_exp_avg_sq,
        param_out=dev_param_out,
        exp_avg_out=dev_exp_avg_out,
        exp_avg_sq_out=dev_exp_avg_sq_out,
        max_exp_avg_sq_out=dev_max_exp_avg_sq_out,
        compute_kernel_config=compute_kernel_config,
    )

    # assert dev_param.shape.with_tile_padding() == ttnn.Shape(model.weight.shape)

    # param_result = dev_param_out.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().to(torch.bfloat16)
    # exp_avg_result = dev_exp_avg_out.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().to(torch.bfloat16)
    # exp_avg_sq_result = dev_exp_avg_sq_out.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().to(torch.bfloat16)
    # if "max_exp_avg_sq" in optimizer_state_dict["state"][0]:
    #     max_exp_avg_sq_result = dev_max_exp_avg_sq_out.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().to(torch.bfloat16)
    # else:
    #     max_exp_avg_sq_result = None

    # rtol = atol = 0.01
    # passing, out = comp_allclose_and_pcc(model.weight, param_result, pcc=0.999, rtol=rtol, atol=atol)
    # logger.debug(f"Out passing (param)={passing}")
    # logger.debug(f"Output pcc={out}")

    # passing, out = comp_allclose_and_pcc(cpu_exp_avg_result, exp_avg_result, pcc=0.999, rtol=rtol, atol=atol)
    # logger.debug(f"Out passing (exp_avg)={passing}")
    # logger.debug(f"Output pcc={out}")

    # passing, out = comp_allclose_and_pcc(cpu_exp_avg_sq_result, exp_avg_sq_result, pcc=0.999, rtol=rtol, atol=atol)
    # logger.debug(f"Out passing (exp_avg_sq)={passing}")
    # logger.debug(f"Output pcc={out}")

    # if "max_exp_avg_sq" in optimizer_state_dict["state"][0]:
    #     passing, out = comp_allclose_and_pcc(
    #         cpu_max_exp_avg_sq_result, max_exp_avg_sq_result, pcc=0.999, rtol=rtol, atol=atol
    #     )
    #     logger.debug(f"Out passing (max_exp_avg_sq)={passing}")
    #     logger.debug(f"Output pcc={out}")
    # assert passing


# @pytest.mark.parametrize(
#     "shape",
#     [
#         [1024, 64, 64],
#         # [2, 2, 2, 2, 2, 2, 64, 64],
#     ],
# )
# @pytest.mark.parametrize("lr", [0.0])
# @pytest.mark.parametrize("betas", [(0.9, 0.999)])
# @pytest.mark.parametrize("eps", [1e-06])
# @pytest.mark.parametrize("weight_decay", [0.3])
# @pytest.mark.parametrize("amsgrad", [True], ids=["AMSGRAD=True"])
# @pytest.mark.parametrize("step", [1000])
# @pytest.mark.parametrize("fp32_dest_acc_en", [True],
#                          ids=["fp32_dest_acc_en=True"])
# @pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16])


@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32_dest_acc_en=True", "fp32_dest_acc_en=False"])
@pytest.mark.parametrize("step", [1, 10, 1000])
@pytest.mark.parametrize("amsgrad", [True, False], ids=["AMSGRAD=True", "AMSGRAD=False"])
@pytest.mark.parametrize("weight_decay", [0.01, 0.3])
@pytest.mark.parametrize("eps", [1e-08, 1e-06])
@pytest.mark.parametrize(
    "betas", [[0.9, 0.999], [0.5, 0.555], [0.855, 0.955]], ids=["[0.9, 0.999]", "[0.5, 0.555]", "[0.855, 0.955]"]
)
@pytest.mark.parametrize("lr", [1e-2, 0.0])
@pytest.mark.parametrize(
    "shape",
    [
        [2, 2, 2, 2, 2, 2, 64, 64],
        [1024, 64, 64],
    ],
    ids=[
        "[2, 2, 2, 2, 2, 2, 64, 64]",
        "[1024, 64, 64]",
    ],
)
def test_moreh_adamw(shape, lr, betas, eps, weight_decay, amsgrad, step, fp32_dest_acc_en, ttnn_dtype, device):
    torch.manual_seed(0)

    iterations = 1
    for i in range(iterations):
        run_moreh_adamw(
            shape, lr, betas, eps, weight_decay, amsgrad, step, fp32_dest_acc_en, device, ttnn_dtype=ttnn_dtype
        )


# @pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16])
# @pytest.mark.parametrize("fp32_dest_acc_en", [
#     True,
#     False
#     ],
#     ids=[
#     "fp32_dest_acc_en=True",
#     "fp32_dest_acc_en=False"
# ])
# @pytest.mark.parametrize("step", [
#     1,
#     10,
#     1000,
# ])
# @pytest.mark.parametrize("amsgrad", [
#     True,
#     False
#     ],
#     ids=[
#         "AMSGRAD=True",
#         "AMSGRAD=False"
# ])
# @pytest.mark.parametrize("weight_decay", [
#     0.01,
#     0.3
#     ])
# @pytest.mark.parametrize("eps", [
#     1e-08,
#     1e-06])
# @pytest.mark.parametrize("betas", [
#     [0.9, 0.999],
#     [0.5, 0.555],
#     [0.855, 0.955]
#     ],
#     ids=[
#     "[0.9, 0.999]",
#     "[0.5, 0.555]",
#     "[0.855, 0.955]"
# ])
# @pytest.mark.parametrize("lr", [
#     1e-2,
#     0.0
#     ])
# @pytest.mark.parametrize(
#     "shape",
#     [
#         [2, 2, 2, 2, 2, 2, 64, 64],
#         [1024, 64, 64],
#     ],
#     ids=["[2, 2, 2, 2, 2, 2, 64, 64]",
#         "[1024, 64, 64]",
#         ]
# )


@pytest.mark.parametrize(
    "shape",
    [
        [420, 64, 64],
        # [2, 2, 2, 2, 2, 2, 64, 64]
    ],
)
@pytest.mark.parametrize("lr", [0.0])
@pytest.mark.parametrize("betas", [(0.9, 0.999)])
@pytest.mark.parametrize("eps", [1e-06])
@pytest.mark.parametrize("weight_decay", [0.3])
@pytest.mark.parametrize("amsgrad", [True], ids=["AMSGRAD=True"])
@pytest.mark.parametrize("step", [1000])
@pytest.mark.parametrize("fp32_dest_acc_en", [True], ids=["fp32_dest_acc_en=True"])
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16])
def test_moreh_adam(shape, lr, betas, eps, weight_decay, amsgrad, step, fp32_dest_acc_en, ttnn_dtype, device):
    torch.manual_seed(0)

    iterations = 10
    for i in range(iterations):
        run_moreh_adam(
            shape, lr, betas, eps, weight_decay, amsgrad, fp32_dest_acc_en, device, dtype=ttnn_dtype, step=step
        )
