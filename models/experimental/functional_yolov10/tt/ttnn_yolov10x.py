# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.experimental.functional_yolov10.tt.common import Conv
from models.experimental.functional_yolov10.reference.yolov10 import Conv as Torch_conv
import math


class ttnn_CIB:
    def __init__(self, shortcut=True, device=None, parameters=None, conv_pt=None, torch_conv=False):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.conv0 = Conv(
            device,
            parameters.cv1[0],
            self.conv_pt.cv1[0],
        )

        self.conv1 = Conv(
            device,
            parameters.cv1[1],
            self.conv_pt.cv1[1],
        )

        if torch_conv:
            self.conv2 = nn.Conv2d(
                in_channels=640,
                out_channels=640,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=640,
                bias=False,
            )
            self.conv2.weight = torch.nn.Parameter(ttnn.to_torch(ttnn.from_device(self.conv_pt.cv1[2].conv.weight)))
            self.conv2.bias = torch.nn.Parameter(
                ttnn.to_torch(ttnn.from_device(self.conv_pt.cv1[2].conv.bias)).squeeze(0).squeeze(0).squeeze(0)
            )

        else:
            self.conv2 = Conv(
                device,
                parameters.cv1[2],
                self.conv_pt.cv1[2],
            )

        self.conv3 = Conv(
            device,
            parameters.cv1[3],
            self.conv_pt.cv1[3],
        )

        self.conv4 = Conv(
            device,
            parameters.cv1[4],
            self.conv_pt.cv1[4],
        )

    def __call__(self, x):
        input_tensor = x
        x = self.conv0(x)
        x = self.conv1(x)

        x = ttnn.to_torch(x)
        x = torch.permute(x, (0, 3, 1, 2)).float()
        x = torch.reshape(x, (1, x.shape[1], int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1]))))
        x = self.conv2(x)
        x = torch.permute(x, (0, 2, 3, 1))
        x = x.reshape(1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3])
        x = ttnn.from_torch(
            x, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        x = ttnn.silu(x)

        x = self.conv3(x)
        x = self.conv4(x)

        x = input_tensor + x
        return x


class ttnn_C2fCIB:
    def __init__(self, shortcut=True, n=3, device=None, parameters=None, conv_pt=None, torch_conv=False):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.shortcut = shortcut

        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
        )
        if torch_conv:
            self.cv2 = nn.Conv2d(
                in_channels=2560,
                out_channels=640,
                kernel_size=1,
                stride=1,
                bias=False,
            )
            self.cv2.weight = torch.nn.Parameter(ttnn.to_torch(ttnn.from_device(self.conv_pt.cv2.conv.weight)))
            self.cv2.bias = torch.nn.Parameter(
                ttnn.to_torch(ttnn.from_device(self.conv_pt.cv2.conv.bias)).squeeze(0).squeeze(0).squeeze(0)
            )
        else:
            self.cv2 = Conv(
                device,
                parameters.cv2,
                self.conv_pt.cv2,
            )

        self.m = [
            ttnn_CIB(
                shortcut=self.shortcut,
                device=self.device,
                parameters=self.parameters[2],
                conv_pt=self.conv_pt.m[_],
                torch_conv=True,
            )
            for _ in range(n)
        ]

    def __call__(self, x):
        x = self.cv1(x)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.permute(x, (0, 3, 1, 2))
        x = ttnn.reshape(x, (1, x.shape[1], int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1]))))
        y = list(ttnn.split(x, 2, 1))

        y[0] = ttnn.permute(y[0], (0, 2, 3, 1))
        y[0] = ttnn.reshape(y[0], (1, 1, y[0].shape[0] * y[0].shape[1] * y[0].shape[2], y[0].shape[3]))
        y[0] = ttnn.to_layout(y[0], ttnn.TILE_LAYOUT)

        y[1] = ttnn.permute(y[1], (0, 2, 3, 1))
        y[1] = ttnn.reshape(y[1], (1, 1, y[1].shape[0] * y[1].shape[1] * y[1].shape[2], y[1].shape[3]))
        y[1] = ttnn.to_layout(y[1], ttnn.TILE_LAYOUT)

        y = [y[0], y[1]]

        for m in self.m:
            out = m(y[-1])

            y.append(ttnn.to_layout(out, ttnn.TILE_LAYOUT))

        out = ttnn.concat(y, -1)

        x = self.cv2(out)

        return x
