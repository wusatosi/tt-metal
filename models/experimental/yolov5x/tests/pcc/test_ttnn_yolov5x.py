import ttnn
import torch
import pytest
from ultralytics import YOLO
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.yolov5x.reference.yolov5x import YOLOv5
from models.experimental.yolov5x.tt.ttnn_yolov5x import (
    Conv,
    Bottleneck,
    SPPF,
    C3,
    Detect,
    Yolov5,
)
from models.experimental.yolov5x.tt.model_preprocessing import (
    create_yolov5x_input_tensors,
    create_yolov5x_model_parameters,
    create_yolov5x_model_parameters_detect,
)


@pytest.mark.parametrize(
    "index, fwd_input_shape, activation",
    [
        (
            0,
            (1, 3, 640, 640),
            "silu",
        ),
        (
            1,
            (1, 80, 320, 320),
            "silu",
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov5x_Conv(
    device,
    use_program_cache,
    reset_seeds,
    index,
    fwd_input_shape,
    activation,
):
    torch_input, ttnn_input = create_yolov5x_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)

    model = YOLO("yolov5xu.pt").eval()
    model = model.get_submodule(f"model.model.{index}")
    state_dict = model.state_dict()

    torch_model = YOLOv5()
    torch_model = torch_model.model.model[index]

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(state_dict.items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    parameters = create_yolov5x_model_parameters(torch_model, torch_input, device=device)

    with torch.inference_mode():
        torch_model_output = torch_model(torch_input)[0]

        ttnn_module = Conv(device, parameters.conv_args, parameters)
        ttnn_output = ttnn_module(ttnn_input)
        ttnn_output = ttnn.to_torch(ttnn_output)
        ttnn_output = ttnn_output.permute(0, 3, 1, 2)
        ttnn_output = ttnn_output.reshape(torch_model_output.shape)

    assert_with_pcc(torch_model_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "index, fwd_input_shape , shortcut",
    [
        (
            2,
            (1, 80, 160, 160),
            True,
        ),
        (
            4,
            (1, 160, 80, 80),
            True,
        ),
        (
            17,
            (1, 160, 80, 80),
            False,
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov5x_Bottleneck(
    device,
    use_program_cache,
    reset_seeds,
    index,
    fwd_input_shape,
    shortcut,
):
    torch_input, ttnn_input = create_yolov5x_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)

    model = YOLO("yolov5xu.pt").eval()
    model = model.get_submodule(f"model.model.{index}.m.{0}")
    state_dict = model.state_dict()

    torch_model = YOLOv5()
    torch_model = torch_model.model.model[index].m[0]

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(state_dict.items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    parameters = create_yolov5x_model_parameters(torch_model, torch_input, device=device)

    torch_model_output = torch_model(torch_input)[0]

    ttnn_module = Bottleneck(shortcut=shortcut, device=device, parameters=parameters.conv_args, conv_pt=parameters)
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_model_output.shape)

    assert_with_pcc(torch_model_output, ttnn_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov5x_SPPF(device, use_program_cache, reset_seeds):
    fwd_input_shape = [1, 1280, 20, 20]
    torch_input, ttnn_input = create_yolov5x_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)

    model = YOLO("yolov5xu.pt").eval()
    model = model.get_submodule(f"model.model.{9}")
    state_dict = model.state_dict()

    torch_model = YOLOv5()
    torch_model = torch_model.model.model[9]

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(state_dict.items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_model_output = torch_model(torch_input)[0]

    parameters = create_yolov5x_model_parameters(torch_model, torch_input, device=device)

    ttnn_module = SPPF(
        device=device,
        parameters=parameters.conv_args,
        conv_pt=parameters,
    )
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)

    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_model_output.shape)

    assert_with_pcc(torch_model_output, ttnn_output, 0.9999)


@pytest.mark.parametrize(
    "index, fwd_input_shape , num_layers, shortcut",
    [
        (
            2,
            (1, 160, 160, 160),
            4,
            True,
        ),
        (
            4,
            (1, 320, 80, 80),
            8,
            True,
        ),
        (
            17,
            (1, 640, 80, 80),
            4,
            False,
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov5x_C3(
    device,
    use_program_cache,
    reset_seeds,
    index,
    fwd_input_shape,
    num_layers,
    shortcut,
):
    torch_input, ttnn_input = create_yolov5x_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)

    model = YOLO("yolov5xu.pt").eval()
    model = model.get_submodule(f"model.model.{index}")
    state_dict = model.state_dict()

    torch_model = YOLOv5()
    torch_model = torch_model.model.model[index]

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(state_dict.items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    parameters = create_yolov5x_model_parameters(torch_model, torch_input, device=device)

    torch_model_output = torch_model(torch_input)[0]

    ttnn_module = C3(
        shortcut=shortcut, n=num_layers, device=device, parameters=parameters.conv_args, conv_pt=parameters
    )
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_model_output.shape)

    assert_with_pcc(torch_model_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "fwd_input_shape",
    [
        ([1, 320, 80, 80], [1, 640, 40, 40], [1, 1280, 20, 20]),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 83000}], indirect=True)
def test_yolov5x_Detect(
    device,
    reset_seeds,
    fwd_input_shape,
):
    torch_input_1, ttnn_input_1 = create_yolov5x_input_tensors(
        device,
        batch_size=fwd_input_shape[0][0],
        input_channels=fwd_input_shape[0][1],
        input_height=fwd_input_shape[0][2],
        input_width=fwd_input_shape[0][3],
    )
    torch_input_2, ttnn_input_2 = create_yolov5x_input_tensors(
        device,
        batch_size=fwd_input_shape[1][0],
        input_channels=fwd_input_shape[1][1],
        input_height=fwd_input_shape[1][2],
        input_width=fwd_input_shape[1][3],
    )
    torch_input_3, ttnn_input_3 = create_yolov5x_input_tensors(
        device,
        batch_size=fwd_input_shape[2][0],
        input_channels=fwd_input_shape[2][1],
        input_height=fwd_input_shape[2][2],
        input_width=fwd_input_shape[2][3],
    )

    torch_input = [torch_input_1, torch_input_2, torch_input_3]

    model = YOLO("yolov5xu.pt").eval()
    model = model.get_submodule(f"model.model.{24}")
    state_dict = model.state_dict()

    torch_model = YOLOv5()
    torch_model = torch_model.model.model[24]

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(state_dict.items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    parameters = create_yolov5x_model_parameters_detect(
        torch_model, torch_input[0], torch_input[1], torch_input[2], device=device
    )

    torch_model_output = torch_model(torch_input)[0]

    ttnn_module = Detect(
        device=device,
        parameters=parameters.model_args,
        conv_pt=parameters,
    )
    ttnn_output = ttnn_module(ttnn_input_1, ttnn_input_2, ttnn_input_3)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_model_output, ttnn_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov5x(device, reset_seeds):
    torch_input, ttnn_input = create_yolov5x_input_tensors(device)

    model = YOLO("yolov5xu.pt").model.eval()
    state_dict = model.state_dict()

    torch_model = YOLOv5()
    torch_model = torch_model.model

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    parameters = create_yolov5x_model_parameters(torch_model, torch_input, device)

    torch_model_output = torch_model(torch_input)[0]
    ttnn_module = Yolov5(
        device=device,
        parameters=parameters,
        conv_pt=parameters,
    )
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_model_output, ttnn_output, 0.99)
