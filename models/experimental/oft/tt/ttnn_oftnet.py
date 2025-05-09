import ttnn
import torch
import torch.nn as nn

from models.experimental.oft.tt.ttnn_resnet import ResNetFeatures, Conv, BasicBlock
from models.experimental.oft.tt.ttnn_oft import OFT


class OftNet:
    def __init__(
        self,
        device,
        parameters,
        conv_pt,
        block,
        layers,
        mean,
        std,
        y_corners,
        topdown_layers=8,
        grid_res=1,
        grid_height=4,
    ):
        self.frontend = ResNetFeatures(device, parameters.frontend, conv_pt.frontend, block, layers)

        self.lat8 = Conv(parameters.lat8, conv_pt.lat8, stride=1, padding=0)
        self.lat16 = Conv(parameters.lat16, conv_pt.lat16, stride=1, padding=0)
        self.lat32 = Conv(parameters.lat32, conv_pt.lat32, stride=1, padding=0)
        self.gn8 = nn.GroupNorm(16, 256)
        self.gn16 = nn.GroupNorm(16, 256)
        self.gn32 = nn.GroupNorm(16, 256)
        self.gn8.weight = nn.Parameter(parameters.bn8.weight)
        self.gn8.bias = nn.Parameter(parameters.bn8.bias)
        self.gn16.weight = nn.Parameter(parameters.bn16.weight)
        self.gn16.bias = nn.Parameter(parameters.bn16.bias)
        self.gn32.weight = nn.Parameter(parameters.bn32.weight)
        self.gn32.bias = nn.Parameter(parameters.bn32.bias)

        self.oft8 = OFT(device, parameters.oft8, y_corners, 1 / 8.0)
        self.oft16 = OFT(device, parameters.oft16, y_corners, 1 / 16.0)
        self.oft32 = OFT(device, parameters.oft32, y_corners, 1 / 32.0)

        self.topdown = [
            BasicBlock(
                device,
                parameters.topdown[i],
                conv_pt.topdown[i],
                256,
                256,
                stride=1,
                height_sharding=False,
            )
            for i in range(topdown_layers)
        ]

        self.head = Conv(parameters.head, conv_pt.head, stride=1, padding=1)

        self.mean = mean
        self.std = std

    def __call__(self, device, image, calib, grid):
        self.mean = ttnn.reshape(self.mean, (3, 1, 1))
        self.std = ttnn.reshape(self.std, (3, 1, 1))
        image = ttnn.subtract(image, self.mean)
        image = ttnn.div(image, self.std)
        image = ttnn.permute(image, (0, 2, 3, 1))
        image = ttnn.to_layout(image, layout=ttnn.ROW_MAJOR_LAYOUT)

        feats8, feats16, feats32 = self.frontend(device, image)

        lat8, out_h, out_w = self.lat8(device, feats8)
        lat8 = ttnn.to_torch(lat8)
        lat8 = lat8.reshape(lat8.shape[0], out_h, out_w, lat8.shape[-1])
        lat8 = torch.permute(lat8, (0, 3, 1, 2))
        lat8 = self.gn8(lat8)
        lat8 = ttnn.from_torch(lat8, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        lat8 = ttnn.relu(lat8)
        # lat8 = ttnn.permute(lat8, [0, 2, 3, 1])

        lat16, out_h, out_w = self.lat16(device, feats16)
        lat16 = ttnn.to_torch(lat16)
        lat16 = lat16.reshape(lat16.shape[0], out_h, out_w, lat16.shape[-1])
        lat16 = torch.permute(lat16, (0, 3, 1, 2))
        lat16 = self.gn16(lat16)
        lat16 = ttnn.from_torch(lat16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        lat16 = ttnn.relu(lat16)
        # lat16 = ttnn.permute(lat16, [0, 2, 3, 1])

        lat32, out_h, out_w = self.lat32(device, feats32)
        lat32 = ttnn.to_torch(lat32)
        lat32 = lat32.reshape(lat32.shape[0], out_h, out_w, lat32.shape[-1])
        lat32 = torch.permute(lat32, (0, 3, 1, 2))
        lat32 = self.gn32(lat32)
        lat32 = ttnn.from_torch(lat32, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        lat32 = ttnn.relu(lat32)
        # lat32 = ttnn.permute(lat32, [0, 2, 3, 1])
        # return lat8, lat16, lat32

        ortho8 = self.oft8(device, lat8, calib, grid)
        ortho16 = self.oft16(device, lat16, calib, grid)
        ortho32 = self.oft32(device, lat32, calib, grid)
        ortho = ttnn.add(ortho8, ortho16)
        ortho = ttnn.add(ortho, ortho32)

        topdown = ortho
        topdown = ttnn.permute(topdown, [0, 2, 3, 1])
        for layer in self.topdown:
            topdown = layer(device, topdown)

        batch, depth, width, _ = topdown.shape

        outputs, out_h, out_w = self.head(device, topdown)
        outputs = ttnn.sharded_to_interleaved(outputs, ttnn.L1_MEMORY_CONFIG)
        outputs = ttnn.permute(outputs, (0, 3, 1, 2))
        outputs = ttnn.reshape(outputs, (batch, -1, 9, depth, width))

        slices = [1, 3, 3, 2]
        start = 0
        parts = []

        for s in slices:
            parts.append(outputs[:, :, start : start + s, :, :])
            start += s
        parts[0] = ttnn.squeeze(parts[0], 2)
        return parts
