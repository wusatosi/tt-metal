# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import math
from models.demos.ufld_v2.ttnn.common import TtnnUFLDV2Conv2D
from models.demos.ufld_v2.ttnn.ttnn_basic_block import TtnnBasicBlock
from models.utility_functions import _nearest_y
from models.demos.ttnn_resnet.tt.ttnn_functional_resnet50_model_utils import get_conv_input_memory_config


def _nearest_32(x):
    return math.ceil(x / 32) * 32


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


class TtnnResnet34:
    def __init__(self, conv_args, conv_pth, device):
        self.conv1_weight_tensor = conv_pth.conv1.weight
        self.conv1_bias_tensor = conv_pth.conv1.bias
        self.conv1_input_channels = self.conv1_weight_tensor.shape[1]
        self.conv1_output_channels = self.conv1_weight_tensor.shape[0]
        print(
            "values are",
            self.conv1_weight_tensor.shape,
            self.conv1_bias_tensor.shape,
            self.conv1_input_channels,
            self.conv1_output_channels,
        )

        input_channels_alignment = 16
        self.conv1_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat8_b,
            activation="relu",
            deallocate_activation=True,
            input_channels_alignment=input_channels_alignment,
            act_block_h_override=0,  # act_block_h_override,
            transpose_shards=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            enable_subblock_padding=False,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            reshard_if_not_optimal=False,
        )
        # self.conv_config = ttnn.Conv2dConfig(
        #     dtype=activation_dtype,
        #     weights_dtype=weights_dtype,
        #     shard_layout=shard_layout,
        #     deallocate_activation=dealloc_act,
        #     enable_act_double_buffer=False,
        #     enable_split_reader=False,
        #     enable_subblock_padding=False,
        #     reshard_if_not_optimal=True,
        #     activation=activation,
        #     input_channels_alignment=8,
        # )
        self.conv1_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            packer_l1_acc=True,
        )
        self.resnet_first_conv_kernel_size = 3
        self.resnet_first_conv_stride = 2
        self.conv1_kernel_size = (4, 4)
        self.conv1_stride = (1, 1)
        self.conv1_padding = (0, 0)
        self.conv1_input_height = 163
        self.conv1_input_width = 403
        self.conv1_output_height = (
            (self.conv1_input_height - self.conv1_kernel_size[0] + 2 * self.conv1_padding[0]) // self.conv1_stride[0]
        ) + 1
        self.conv1_output_width = (
            (self.conv1_input_width - self.conv1_kernel_size[1] + 2 * self.conv1_padding[1]) // self.conv1_stride[1]
        ) + 1
        print("conv1 params are", self.conv1_output_height, self.conv1_output_width)
        self.fold_stride_h = self.resnet_first_conv_stride
        self.fold_stride_w = self.resnet_first_conv_stride
        _, c, h, w = (1, 3, 320, 800)
        n = 1
        h += self.resnet_first_conv_kernel_size * 2
        w += self.resnet_first_conv_kernel_size * 2
        C = _nearest_y(c, 4)
        self.fold_pad_c = C - c
        self.fold_pad_h = self.resnet_first_conv_kernel_size
        self.fold_pad_w = self.resnet_first_conv_kernel_size
        self.fold_output_shape = (
            n,
            h // self.fold_stride_h,
            w // self.fold_stride_w,
            C * (self.fold_stride_h * self.fold_stride_w),
        )
        print("folding params", self.fold_pad_c, self.fold_pad_h, self.fold_pad_w, self.fold_output_shape)
        num_cores_x = 8
        num_cores_y = 8
        self.fold_compute_grid_size = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
        )
        conv_dummy_tensor = torch.rand((self.fold_output_shape), dtype=torch.bfloat16)
        conv_dummy_tensor = ttnn.from_torch(conv_dummy_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.batch_size = 1
        self.override_fold_mem_config = get_conv_input_memory_config(
            self.batch_size,
            self.conv1_input_channels,
            self.conv1_input_height,
            self.conv1_input_width,
            self.conv1_output_channels,
            self.conv1_output_height,
            self.conv1_output_width,
            device.compute_with_storage_grid_size(),
            self.conv1_config.input_channels_alignment,
            False,
        )

        # old
        self.maxpool_args = conv_args.maxpool
        self.device = device
        self.conv1 = TtnnUFLDV2Conv2D(
            conv_args.conv1, conv_pth.conv1, device=self.device, activation="relu", dealloc_act=True
        )
        self.layer1_0 = TtnnBasicBlock(conv_args.layer1[0], conv_pth.layer1_0, device=self.device, is_downsample=False)
        self.layer1_1 = TtnnBasicBlock(conv_args.layer1[1], conv_pth.layer1_1, device=self.device, is_downsample=False)
        self.layer1_2 = TtnnBasicBlock(conv_args.layer1[2], conv_pth.layer1_2, device=self.device, is_downsample=False)
        self.layer2_0 = TtnnBasicBlock(conv_args.layer2[0], conv_pth.layer2_0, device=self.device, is_downsample=True)
        self.layer2_1 = TtnnBasicBlock(conv_args.layer2[1], conv_pth.layer2_1, device=self.device, is_downsample=False)
        self.layer2_2 = TtnnBasicBlock(conv_args.layer2[2], conv_pth.layer2_2, device=self.device, is_downsample=False)
        self.layer2_3 = TtnnBasicBlock(conv_args.layer2[3], conv_pth.layer2_3, device=self.device, is_downsample=False)
        self.layer3_0 = TtnnBasicBlock(
            conv_args.layer3[0], conv_pth.layer3_0, device=self.device, is_downsample=True, blk_sharded=True
        )
        self.layer3_1 = TtnnBasicBlock(
            conv_args.layer3[1], conv_pth.layer3_1, device=self.device, is_downsample=False, blk_sharded=True
        )
        self.layer3_2 = TtnnBasicBlock(
            conv_args.layer3[2], conv_pth.layer3_2, device=self.device, is_downsample=False, blk_sharded=True
        )
        self.layer3_3 = TtnnBasicBlock(
            conv_args.layer3[3], conv_pth.layer3_3, device=self.device, is_downsample=False, blk_sharded=True
        )
        self.layer3_4 = TtnnBasicBlock(
            conv_args.layer3[4], conv_pth.layer3_4, device=self.device, is_downsample=False, blk_sharded=True
        )
        self.layer3_5 = TtnnBasicBlock(
            conv_args.layer3[5], conv_pth.layer3_5, device=self.device, is_downsample=False, blk_sharded=True
        )
        self.layer4_0 = TtnnBasicBlock(
            conv_args.layer4[0], conv_pth.layer4_0, device=self.device, is_downsample=True, blk_sharded=True
        )
        self.layer4_1 = TtnnBasicBlock(
            conv_args.layer4[1], conv_pth.layer4_1, device=self.device, is_downsample=False, blk_sharded=True
        )
        self.layer4_2 = TtnnBasicBlock(
            conv_args.layer4[2], conv_pth.layer4_2, device=self.device, is_downsample=False, blk_sharded=True
        )

    def __call__(self, x, batch_size=1):
        p(x, "input_tensor")
        print("config is", self.override_fold_mem_config)
        fold_output_tensor = ttnn.fold(
            x,
            self.fold_stride_h,
            self.fold_stride_w,
            use_transpose_as_fold=True,
            pad_c=self.fold_pad_c,
            pad_h=self.fold_pad_h,
            pad_w=self.fold_pad_w,
            grid_size=self.fold_compute_grid_size,
            override_memory_config=self.override_fold_mem_config,
        )
        p(fold_output_tensor, "fold_output_tensor")
        n, c, h, w = fold_output_tensor.shape
        fold_output_tensor = ttnn.reshape(fold_output_tensor, (1, 1, n * c * h, w))
        p(fold_output_tensor, "fold_output_tensor after reshape")
        ttnn.deallocate(x)
        ss

        # old
        x1, out_ht, out_wdth = self.conv1(x)
        x1 = ttnn.max_pool2d(
            x1,
            batch_size=batch_size,
            input_h=out_ht,
            input_w=out_wdth,
            channels=x.shape[-1],
            kernel_size=[self.maxpool_args.kernel_size, self.maxpool_args.kernel_size],
            stride=[self.maxpool_args.stride, self.maxpool_args.stride],
            padding=[self.maxpool_args.padding, self.maxpool_args.padding],
            dilation=[self.maxpool_args.dilation, self.maxpool_args.dilation],
        )
        x = ttnn.sharded_to_interleaved(x1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x1)
        x = ttnn.reallocate(x)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = self.layer1_0(x)
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        x = self.layer2_0(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)
        x = self.layer3_0(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        x = self.layer3_4(x)
        x = self.layer3_5(x)
        x = self.layer4_0(x)
        x = self.layer4_1(x)
        x = self.layer4_2(x)

        return x
