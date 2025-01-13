# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import ttnn
import math
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

transform = transforms.ToTensor()
mnist_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

image1, _ = mnist_dataset[0]
image2, _ = mnist_dataset[1]

image1_pil = transforms.ToPILImage()(image1)
image2_pil = transforms.ToPILImage()(image2)

image1_path = "mnist_image1.png"
image2_path = "mnist_image2.png"

image1_pil.save(image1_path)
image2_pil.save(image2_path)

MNIST_INPUT_FEATURE_SIZE = 784
MNIST_OUTPUT_FEATURE_SIZE = 10
MNIIST_HIDDEN_SIZE = 256


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
        mem_config = ttnn.create_sharded_memory_config(
            shape=self.l1_weight.shape.with_tile_padding(),
            core_grid=ttnn.CoreGrid(x=8, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
        )

        self.l1_weight = ttnn.to_memory_config(self.l1_weight, mem_config)

        self.l1_bias = l1_bias
        self.l1_bias = ttnn.to_memory_config(self.l1_bias, mem_config)
        self.l2_weight = l2_weight
        self.l2_bias = l2_bias

        self.mem_config1 = ttnn.create_sharded_memory_config(
            shape=(32, 256), core_grid=ttnn.CoreGrid(x=8, y=1), strategy=ttnn.ShardStrategy.WIDTH
        )

        self.prog = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 1),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,  # M // 32
            per_core_N=1,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def __call__(self, x):
        output = ttnn.linear(
            x,
            self.l1_weight,
            bias=self.l1_bias,
            memory_config=self.mem_config1,
            dtype=ttnn.bfloat16,
            program_config=self.prog,
        )
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


def do_inference_ttnn(model_ttnn, device, torch_input):
    torch_input = torch_input.flatten(start_dim=1)

    torch_input = myf(torch_input)

    input_ttnn = ttnn.from_torch(
        torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    output_ttnn = model_ttnn(input_ttnn)

    output_ttnn = ttnn.untilize_with_unpadding(output_ttnn, (0, 9))


def do_inference_torch(model_torch, torch_input):
    torch_input = torch_input.flatten(start_dim=1)

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

    model_ttnn = MyMNIST(
        ttnn.tilize_with_val_padding(
            ttnn.from_torch(l1_weightT, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG),
            output_tensor_shape=fpadded_shape(l1_weightT),
            pad_value=0,
        ),
        ttnn.tilize_with_val_padding(
            ttnn.from_torch(l1_bias, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG),
            output_tensor_shape=fpadded_shape(l1_bias),
            pad_value=0,
        ),
        ttnn.tilize_with_val_padding(
            ttnn.from_torch(l2_weightT, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG),
            output_tensor_shape=fpadded_shape(l2_weightT),
            pad_value=0,
        ),
        ttnn.tilize_with_val_padding(
            ttnn.from_torch(l2_bias, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG),
            output_tensor_shape=fpadded_shape(l2_bias),
            pad_value=0,
        ),
    )

    training_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=32, shuffle=False)

    for _ in range(2):
        inputs, _ = next(iter(training_loader))
        do_inference_torch(model_torch, inputs)
        do_inference_ttnn(model_ttnn, device, inputs)

    ttnn.DumpDeviceProfiler(device)  # important for device profiling
    ttnn.close_device(device)


test_mnist()
