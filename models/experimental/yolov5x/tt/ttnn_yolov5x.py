import ttnn
import math
from models.experimental.yolo_common.yolo_utils import concat, determine_num_cores, get_core_grid_from_num_cores


def deallocate_tensors(*tensors):
    for t in tensors:
        ttnn.deallocate(t)


def interleaved_to_sharded(x):
    x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
    nhw = x.shape[0] * x.shape[1] * x.shape[2]
    num_cores = determine_num_cores(nhw, int(math.sqrt(x.shape[2])))
    num_cores = 56 if num_cores == 1 else num_cores
    core_grid = get_core_grid_from_num_cores(num_cores)
    shardspec = ttnn.create_sharded_memory_config_(
        x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )

    return ttnn.reshard(x, shardspec) if x.is_sharded() else ttnn.interleaved_to_sharded(x, shardspec)


class Conv:
    def __init__(
        self,
        device,
        conv,
        conv_pth,
        bn=None,
        activation="",
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        use_1d_systolic_array=True,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        is_detect=False,
        is_dfl=False,
        config_override=None,
        auto_shard=False,
        deallocate_activation=False,
    ):
        self.is_detect = is_detect
        self.is_dfl = is_dfl
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.use_1d_systolic_array = use_1d_systolic_array
        self.deallocate_activation = deallocate_activation
        self.auto_shard = auto_shard
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=False,
        )

        if shard_layout is None and not self.auto_shard:
            shard_layout = (
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                if use_1d_systolic_array
                else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            )

        self.conv_config = ttnn.Conv2dConfig(
            dtype=activation_dtype,
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=self.deallocate_activation,
            input_channels_alignment=32,
            reshard_if_not_optimal=True if self.use_1d_systolic_array else False,
            activation=activation,
            enable_subblock_padding=False,
            output_layout=ttnn.TILE_LAYOUT,
        )
        if auto_shard:
            self.conv_config.shard_layout = None

        config_override = None
        config_override = {"act_block_h": 64} if conv.in_channels == 3 or conv.in_channels == 6 else None
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]

        if "bias" in conv_pth:
            bias = ttnn.from_device(conv_pth.bias)
            self.bias = bias
        else:
            self.bias = None

        weight = ttnn.from_device(conv_pth.weight)
        self.weight = weight

    def __call__(self, x):
        if self.is_detect:
            input_height = int(math.sqrt(x.shape[2]))
            input_width = int(math.sqrt(x.shape[2]))
            batch_size = x.shape[0]
        elif self.is_dfl:
            input_height = x.shape[1]
            input_width = x.shape[2]
            batch_size = x.shape[0]
        else:
            batch_size = self.conv.batch_size
            input_height = self.conv.input_height
            input_width = self.conv.input_width

        [x, [output_height, output_width]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=input_height,
            input_width=input_width,
            batch_size=batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            conv_config=self.conv_config,
            groups=self.groups,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=False,
        )
        return x


class Bottleneck:
    def __init__(self, shortcut=True, device=None, parameters=None, conv_pt=None, label=None):
        self.shortcut = shortcut
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.label = label

        self.cv1 = Conv(
            device,
            parameters.cv1.conv,
            self.conv_pt.cv1.conv,
            activation="silu",
        )

        self.cv2 = Conv(
            device,
            parameters.cv2.conv,
            self.conv_pt.cv2.conv,
            auto_shard=True,
            activation="silu",
        )

    def __call__(self, input_tensor):
        cv1 = self.cv1(input_tensor)

        cv1 = ttnn.sharded_to_interleaved(cv1, memory_config=ttnn.L1_MEMORY_CONFIG)

        cv1 = self.cv2(cv1)

        cv1 = ttnn.sharded_to_interleaved(cv1, memory_config=ttnn.L1_MEMORY_CONFIG)

        if self.label:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
        return ttnn.add(input_tensor, cv1, memory_config=ttnn.L1_MEMORY_CONFIG) if self.shortcut else cv1


class SPPF:
    def __init__(self, device=None, parameters=None, conv_pt=None):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt

        self.cv1 = Conv(
            device,
            parameters.cv1.conv,
            self.conv_pt.cv1.conv,
            activation="silu",
            use_1d_systolic_array=False,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )

        self.cv2 = Conv(
            device,
            parameters.cv2.conv,
            self.conv_pt.cv2.conv,
            auto_shard=True,
            activation="silu",
        )

    def __call__(self, x):
        cv1 = self.cv1(x)
        cv1 = ttnn.sharded_to_interleaved(cv1, memory_config=ttnn.L1_MEMORY_CONFIG)
        cv1 = ttnn.to_layout(cv1, ttnn.ROW_MAJOR_LAYOUT)
        y = [cv1]

        TILE_WIDTH = 32
        in_c = self.parameters.cv2.conv.in_channels
        in_c_padded = in_c
        if in_c % TILE_WIDTH != 0 and in_c != 16:
            in_c_padded = in_c + (TILE_WIDTH - in_c % TILE_WIDTH)

        for i in range(3):
            if y[-1].is_sharded():
                y[-1] = ttnn.sharded_to_interleaved(y[-1])
            tt_out = ttnn.max_pool2d(
                input_tensor=y[-1],
                batch_size=x.shape[0],
                input_h=20,
                input_w=20,
                channels=in_c_padded,
                kernel_size=[5, 5],
                stride=[1, 1],
                padding=[2, 2],
                dilation=[1, 1],
                applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            )
            y.append(tt_out)

        out = concat(-1, True, *y)

        deallocate_tensors(*y)

        out = ttnn.sharded_to_interleaved(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        out = self.cv2(out)

        return out


class C3:
    def __init__(self, shortcut=True, n=4, device=None, parameters=None, conv_pt=None):
        self.shortcut = shortcut
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt

        self.cv1 = Conv(
            device,
            parameters.cv1.conv,
            self.conv_pt.cv1.conv,
            activation="silu",
        )

        self.cv2 = Conv(
            device,
            parameters.cv2.conv,
            self.conv_pt.cv2.conv,
            activation="silu",
        )

        self.cv3 = Conv(
            device,
            parameters.cv3.conv,
            self.conv_pt.cv3.conv,
            activation="silu",
            auto_shard=True,
        )

        self.m = [
            Bottleneck(
                self.shortcut,
                device=self.device,
                parameters=self.parameters.m[i],
                conv_pt=self.conv_pt.m[i],
                label=(i == 0),
            )
            for i in range(n)
        ]

    def __call__(self, input_tensor):
        m_out = self.cv1(input_tensor)

        for m in self.m:
            m_out = m(m_out)

        cv2_out = self.cv2(input_tensor)

        if cv2_out.shape[2] != m_out.shape[2]:
            cv2_out = ttnn.sharded_to_interleaved(cv2_out, memory_config=ttnn.L1_MEMORY_CONFIG)
            cv2_out = cv2_out[:, :, : m_out.shape[2], :]
            cv2_out = interleaved_to_sharded(cv2_out)

        concat_out = concat(-1, True, m_out, cv2_out)
        concat_out = ttnn.sharded_to_interleaved(concat_out, memory_config=ttnn.L1_MEMORY_CONFIG)

        out = self.cv3(concat_out)
        deallocate_tensors(m_out, cv2_out)

        return out


class Detect:
    def __init__(self, device=None, parameters=None, conv_pt=None):
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.device = device

        self.cv2_0_0 = Conv(
            device, parameters.cv2[0][0].conv, self.conv_pt.cv2[0][0].conv, is_detect=True, activation="silu"
        )
        self.cv2_0_1 = Conv(
            device, parameters.cv2[0][1].conv, self.conv_pt.cv2[0][1].conv, is_detect=True, activation="silu"
        )
        self.cv2_0_2 = Conv(device, parameters.cv2[0][2], self.conv_pt.cv2[0][2], is_detect=True)

        self.cv2_1_0 = Conv(
            device, parameters.cv2[1][0].conv, self.conv_pt.cv2[1][0].conv, is_detect=True, activation="silu"
        )
        self.cv2_1_1 = Conv(
            device, parameters.cv2[1][1].conv, self.conv_pt.cv2[1][1].conv, is_detect=True, activation="silu"
        )
        self.cv2_1_2 = Conv(device, parameters.cv2[1][2], self.conv_pt.cv2[1][2], is_detect=True)

        self.cv2_2_0 = Conv(
            device, parameters.cv2[2][0].conv, self.conv_pt.cv2[2][0].conv, is_detect=True, activation="silu"
        )
        self.cv2_2_1 = Conv(
            device, parameters.cv2[2][1].conv, self.conv_pt.cv2[2][1].conv, is_detect=True, activation="silu"
        )
        self.cv2_2_2 = Conv(device, parameters.cv2[2][2], self.conv_pt.cv2[2][2], is_detect=True)

        self.cv3_0_0 = Conv(
            device, parameters.cv3[0][0].conv, self.conv_pt.cv3[0][0].conv, is_detect=True, activation="silu"
        )
        self.cv3_0_1 = Conv(
            device, parameters.cv3[0][1].conv, self.conv_pt.cv3[0][1].conv, is_detect=True, activation="silu"
        )
        self.cv3_0_2 = Conv(device, parameters.cv3[0][2], self.conv_pt.cv3[0][2], is_detect=True)

        self.cv3_1_0 = Conv(
            device, parameters.cv3[1][0].conv, self.conv_pt.cv3[1][0].conv, is_detect=True, activation="silu"
        )
        self.cv3_1_1 = Conv(
            device, parameters.cv3[1][1].conv, self.conv_pt.cv3[1][1].conv, is_detect=True, activation="silu"
        )
        self.cv3_1_2 = Conv(device, parameters.cv3[1][2], self.conv_pt.cv3[1][2], is_detect=True)

        self.cv3_2_0 = Conv(
            device,
            parameters.cv3[2][0].conv,
            self.conv_pt.cv3[2][0].conv,
            is_detect=True,
            activation="silu",
            use_1d_systolic_array=False,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )
        self.cv3_2_1 = Conv(
            device, parameters.cv3[2][1].conv, self.conv_pt.cv3[2][1].conv, is_detect=True, activation="silu"
        )
        self.cv3_2_2 = Conv(device, parameters.cv3[2][2], self.conv_pt.cv3[2][2], is_detect=True)

        self.dfl = Conv(device, parameters.dfl.conv, self.conv_pt.dfl.conv, is_dfl=True)

        self.anchors = conv_pt.anchors
        self.strides = conv_pt.strides

    def __call__(self, y1, y2, y3):
        x1 = self.cv2_0_0(y1)
        x1 = self.cv2_0_1(x1)
        x1 = self.cv2_0_2(x1)
        x2 = self.cv2_1_0(y2)
        x2 = self.cv2_1_1(x2)
        x2 = self.cv2_1_2(x2)
        x3 = self.cv2_2_0(y3)
        x3 = self.cv2_2_1(x3)
        x3 = self.cv2_2_2(x3)
        x4 = self.cv3_0_0(y1)
        x4 = self.cv3_0_1(x4)
        x4 = self.cv3_0_2(x4)
        y1 = concat(-1, False, x1, x4)
        ttnn.deallocate(x1)
        ttnn.deallocate(x4)
        x5 = self.cv3_1_0(y2)
        x5 = self.cv3_1_1(x5)
        x5 = self.cv3_1_2(x5)
        y2 = concat(-1, False, x2, x5)
        ttnn.deallocate(x2)
        ttnn.deallocate(x5)
        x6 = self.cv3_2_0(y3)
        x6 = self.cv3_2_1(x6)
        x6 = self.cv3_2_2(x6)

        y3 = concat(-1, False, x3, x6)
        ttnn.deallocate(x3)
        ttnn.deallocate(x6)

        y = concat(2, False, y1, y2, y3)

        ttnn.deallocate(y1)
        ttnn.deallocate(y2)
        ttnn.deallocate(y3)

        ya, yb = y[:, :, :, :64], y[:, :, :, 64:144]
        ya = ttnn.permute(ya, (0, 1, 3, 2))
        ya = ttnn.reshape(ya, (ya.shape[0], 4, 16, ya.shape[-1]))
        ya = ttnn.permute(ya, (0, 2, 1, 3))
        ya = ttnn.to_layout(ya, ttnn.TILE_LAYOUT)
        ya = ttnn.softmax(ya, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ya = ttnn.permute(ya, (0, 2, 3, 1))
        c = self.dfl(ya)

        if c.is_sharded():
            c = ttnn.sharded_to_interleaved(c, memory_config=ttnn.L1_MEMORY_CONFIG)
        c = ttnn.to_layout(c, layout=ttnn.ROW_MAJOR_LAYOUT)
        c = ttnn.permute(c, (0, 3, 1, 2))
        c = ttnn.reshape(c, (c.shape[0], 1, 4, int(c.shape[3] / 4)))
        c = ttnn.reshape(c, (c.shape[0], c.shape[1] * c.shape[2], c.shape[3]))
        c1, c2 = c[:, :2, :], c[:, 2:4, :]

        c1 = ttnn.to_layout(c1, layout=ttnn.TILE_LAYOUT)
        c2 = ttnn.to_layout(c2, layout=ttnn.TILE_LAYOUT)

        c1 = self.anchors - c1
        c2 = self.anchors + c2
        z1 = c2 - c1
        z2 = c1 + c2
        z2 = ttnn.div(z2, 2)

        ttnn.deallocate(ya)
        ttnn.deallocate(c)
        ttnn.deallocate(c1)
        ttnn.deallocate(c2)

        z = concat(1, False, z2, z1)
        z = ttnn.multiply(z, self.strides, memory_config=ttnn.L1_MEMORY_CONFIG)
        yb = ttnn.squeeze(yb, 0)
        yb = ttnn.permute(yb, (0, 2, 1))
        yb = ttnn.sigmoid(yb)

        out = concat(1, False, z, yb)

        ttnn.deallocate(z)
        ttnn.deallocate(yb)

        return out


class Yolov5:
    def __init__(self, device, parameters, conv_pt):
        self.device = device

        self.conv1 = Conv(
            device,
            parameters.conv_args[0].conv,
            conv_pt.model[0].conv,
            config_override={"act_block_h": 64},
            activation="silu",
        )
        self.conv2 = Conv(
            device, parameters.conv_args[1].conv, conv_pt.model[1].conv, deallocate_activation=True, activation="silu"
        )
        self.c3_1 = C3(
            shortcut=True, n=4, device=self.device, parameters=parameters.conv_args[2], conv_pt=conv_pt.model[2]
        )
        self.conv3 = Conv(
            device, parameters.conv_args[3].conv, conv_pt.model[3].conv, deallocate_activation=True, activation="silu"
        )
        self.c3_2 = C3(
            shortcut=True, n=8, device=self.device, parameters=parameters.conv_args[4], conv_pt=conv_pt.model[4]
        )
        self.conv4 = Conv(
            device, parameters.conv_args[5].conv, conv_pt.model[5].conv, deallocate_activation=True, activation="silu"
        )
        self.c3_3 = C3(
            shortcut=True, n=8, device=self.device, parameters=parameters.conv_args[6], conv_pt=conv_pt.model[6]
        )
        self.conv5 = Conv(
            device,
            parameters.conv_args[7].conv,
            conv_pt.model[7].conv,
            deallocate_activation=True,
            activation="silu",
            use_1d_systolic_array=False,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )
        self.c3_4 = C3(
            shortcut=True, n=4, device=self.device, parameters=parameters.conv_args[8], conv_pt=conv_pt.model[8]
        )
        self.sppf = SPPF(device, parameters.conv_args[9], conv_pt.model[9])
        self.conv6 = Conv(
            device,
            parameters.conv_args[10].conv,
            conv_pt.model[10].conv,
            deallocate_activation=True,
            activation="silu",
            use_1d_systolic_array=False,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )

        self.c3_5 = C3(
            shortcut=False, n=4, device=self.device, parameters=parameters.conv_args[13], conv_pt=conv_pt.model[13]
        )
        self.conv7 = Conv(
            device, parameters.conv_args[14].conv, conv_pt.model[14].conv, deallocate_activation=True, activation="silu"
        )

        self.c3_6 = C3(
            shortcut=False, n=4, device=self.device, parameters=parameters.conv_args[17], conv_pt=conv_pt.model[17]
        )
        self.conv8 = Conv(
            device, parameters.conv_args[18].conv, conv_pt.model[18].conv, deallocate_activation=True, activation="silu"
        )

        self.c3_7 = C3(
            shortcut=False, n=4, device=self.device, parameters=parameters.conv_args[20], conv_pt=conv_pt.model[20]
        )
        self.conv9 = Conv(
            device,
            parameters.conv_args[21].conv,
            conv_pt.model[21].conv,
            deallocate_activation=True,
            activation="silu",
            use_1d_systolic_array=False,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )

        self.c3_8 = C3(
            shortcut=False, n=4, device=self.device, parameters=parameters.conv_args[23], conv_pt=conv_pt.model[23]
        )
        self.detect = Detect(device, parameters.model_args.model[24], conv_pt.model[24])

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c3_1(x)
        x = self.conv3(x)
        x = self.c3_2(x)
        x4 = x
        x4 = ttnn.reallocate(x4, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.conv4(x)
        x = self.c3_3(x)
        x6 = x
        x6 = ttnn.reallocate(x6, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.conv5(x)
        x = self.c3_4(x)
        x = self.sppf(x)
        x = self.conv6(x)
        x10 = x
        x10 = ttnn.sharded_to_interleaved(x10, memory_config=ttnn.L1_MEMORY_CONFIG)
        x10 = ttnn.reallocate(x10, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = interleaved_to_sharded(x)
        x = ttnn.upsample(x, scale_factor=2)

        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))

        x = concat(-1, True, x, x6)
        ttnn.deallocate(x6)

        x = self.c3_5(x)
        x = self.conv7(x)
        x14 = x
        x14 = ttnn.sharded_to_interleaved(x14, memory_config=ttnn.L1_MEMORY_CONFIG)
        x14 = ttnn.reallocate(x14, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = interleaved_to_sharded(x)
        x = ttnn.upsample(x, scale_factor=2)

        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))

        x = concat(-1, False, x, x4)
        ttnn.deallocate(x4)

        x = self.c3_6(x)
        x17 = x
        x17 = ttnn.reallocate(x17, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.conv8(x)

        x = concat(-1, True, x, x14)
        ttnn.deallocate(x14)

        x = self.c3_7(x)
        x20 = x
        x20 = ttnn.reallocate(x20, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.conv9(x)

        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = concat(-1, True, x, x10)
        ttnn.deallocate(x10)

        x = self.c3_8(x)
        x23 = x
        x23 = ttnn.reallocate(x23, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.detect(x17, x20, x23)

        ttnn.deallocate(x17)
        ttnn.deallocate(x20)
        ttnn.deallocate(x23)

        return x
