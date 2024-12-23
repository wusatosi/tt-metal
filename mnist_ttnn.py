# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import ttnn
import math
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import time

from tracy import Profiler

transform = transforms.ToTensor()  # Transform the image to a tensor
mnist_dataset = MNIST(root="./data", train=True, download=True, transform=transform)

# Batch size configurations
MNIST_BATCH_SIZE_EXP_RANGE = 7

# Input size configurations
MNIIST_INPUT_SIZE_EXP_RANGE = [5, 7]
MNIIST_INPUT_SIZE_FACTORS = [1, 3, 5, 7]

# Hidden layer size configurations
MNIST_HIDDEN_SIZE_EXP_RANGE = [5, 7]
MNIIST_HIDDEN_SIZE_FACTORS = [1, 3]

MNIST_INPUT_FEATURE_SIZE = 784  # 784 = 28 * 28, default size of MNIST image
MNIST_OUTPUT_FEATURE_SIZE = 10  # 10 classes in MNIST, default output size
MNIIST_HIDDEN_SIZE = 256  # Hidden layer size, default size

BATCH_SIZE = [
    2**i for i in range(MNIST_BATCH_SIZE_EXP_RANGE)
]  # Batch size, sizes will be 1, 2, 4, 8, 16, 32, 64, etc.
INPUT_SIZE = [  # Input size, sizes will be 1 * 2^5 = 32, 3 * 2^5 = 96, 5 * 2^5 = 160, 7 * 2^5 = 224, etc.
    factor * hidden
    for factor in MNIIST_INPUT_SIZE_FACTORS
    for hidden in [2**i for i in range(MNIIST_INPUT_SIZE_EXP_RANGE[0], MNIIST_INPUT_SIZE_EXP_RANGE[1])]
]
HIDDEN_SIZE = [  # Hidden layer size, sizes will be 1 * 2^5 = 32, 3 * 2^5 = 96, 1 * 2^6 = 64, 3 * 2^6 = 192, etc.
    factor * hidden
    for factor in MNIIST_HIDDEN_SIZE_FACTORS
    for hidden in [2**i for i in range(MNIST_HIDDEN_SIZE_EXP_RANGE[0], MNIST_HIDDEN_SIZE_EXP_RANGE[1])]
]
ARCH = []
DATAFORMAT = []
MATH_FIDELITY = []


# Model definition
class MNISTLinear(nn.Module):
    def __init__(
        self, input_size=MNIST_INPUT_FEATURE_SIZE, output_size=MNIST_OUTPUT_FEATURE_SIZE, hidden_size=MNIIST_HIDDEN_SIZE
    ):
        super(MNISTLinear, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)

        return nn.functional.softmax(x)


def _nearest_32(x):
    return math.ceil(x / 32) * 32


class MyMNIST:
    def __init__(self, l1_weight, l1_bias, l2_weight, l2_bias):
        self.l1_weight = l1_weight
        self.l1_bias = l1_bias
        self.l2_weight = l2_weight
        self.l2_bias = l2_bias

    def __call__(self, x):
        output = ttnn.matmul(x, self.l1_weight)
        output = output + self.l1_bias
        output = ttnn.relu(output)
        output = ttnn.matmul(output, self.l2_weight)
        output = output + self.l2_bias
        return ttnn.softmax(output)


def myf(x):
    if x.dim() == 1:
        x = x.unsqueeze(0)

    return x


def fpadded_shape(x):
    l = [x.shape[i] for i in range(x.dim() - 2)]
    for i in range(max(x.dim() - 2, 0), x.dim()):
        l.append(_nearest_32(x.shape[i]))

    return l


def do_inference_ttnn(i, model_ttnn, device):
    torch_input, label = mnist_dataset[i]
    torch_input = torch_input.flatten()

    torch_input = myf(torch_input)

    input_ttnn = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    mem_config1 = ttnn.create_sharded_memory_config(
        shape=torch_input.shape, core_grid=ttnn.CoreGrid(x=1, y=1), strategy=ttnn.ShardStrategy.BLOCK
    )

    input_ttnn = ttnn.to_memory_config(input_ttnn, mem_config1)
    # input_ttnn = ttnn.tilize_with_val_padding(ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16), output_tensor_shape=fpadded_shape(torch_input), pad_value=0, memory_config=mem_config1)

    print("input_ttnn shape:", input_ttnn.shape)
    print("input_ttnn layout:", input_ttnn.layout)

    output_ttnn = model_ttnn(input_ttnn)


def do_inference_torch(i, model_torch):
    torch_input, label = mnist_dataset[i]
    torch_input = torch_input.flatten()

    torch_input = myf(torch_input)

    output_torch = model_torch(torch_input)


def test_mnist():
    model_torch = MNISTLinear()
    model_torch.load_state_dict(torch.load("model.pth"))
    device_id = 0
    device = ttnn.open_device(device_id=device_id)
    ttnn.enable_program_cache(device)

    dict = model_torch.state_dict()
    l1_weight = dict["l1.weight"]
    l1_weightT = torch.transpose(l1_weight, 0, 1)
    l1_bias = dict["l1.bias"]

    l2_weight = dict["l2.weight"]
    l2_weightT = torch.transpose(l2_weight, 0, 1)
    l2_bias = dict["l2.bias"]

    l1_weight = myf(l1_weight)
    l1_bias = myf(l1_bias)
    l2_weight = myf(l2_weight)
    l2_bias = myf(l2_bias)

    # print(f'l1_weight shape: {l1_weight.shape}')
    print(f"l1_bias shape: {l1_bias.shape}")
    # print(f'l2_weight shape: {l2_weight.shape}')
    print(f"l2_bias shape: {l2_bias.shape}")

    mem_config2 = ttnn.create_sharded_memory_config(
        shape=l1_bias.shape, core_grid=ttnn.CoreGrid(x=4, y=1), strategy=ttnn.ShardStrategy.WIDTH
    )
    mem_config4 = ttnn.create_sharded_memory_config(
        shape=l2_bias.shape, core_grid=ttnn.CoreGrid(x=5, y=1), strategy=ttnn.ShardStrategy.WIDTH
    )

    model_ttnn = MyMNIST(
        ttnn.tilize_with_val_padding(
            ttnn.from_torch(l1_weightT, device=device, dtype=ttnn.bfloat16),
            output_tensor_shape=fpadded_shape(l1_weightT),
            pad_value=0,
        ),
        ttnn.tilize_with_val_padding(
            ttnn.from_torch(l1_bias, device=device, dtype=ttnn.bfloat16),
            output_tensor_shape=fpadded_shape(l1_bias),
            pad_value=0,
            memory_config=mem_config2,
        ),
        ttnn.tilize_with_val_padding(
            ttnn.from_torch(l2_weightT, device=device, dtype=ttnn.bfloat16),
            output_tensor_shape=fpadded_shape(l2_weightT),
            pad_value=0,
        ),
        ttnn.tilize_with_val_padding(
            ttnn.from_torch(l2_bias, device=device, dtype=ttnn.bfloat16),
            output_tensor_shape=fpadded_shape(l2_bias),
            pad_value=0,
            memory_config=mem_config4,
        ),
    )

    for i in range(2):
        do_inference_ttnn(i, model_ttnn, device)
        do_inference_torch(i, model_torch)

    ttnn.close_device(device)


test_mnist()
