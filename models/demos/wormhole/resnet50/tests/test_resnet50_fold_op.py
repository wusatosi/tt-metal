from models.demos.wormhole.resnet50.demo.demo import test_demo_sample
import pytest


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_loc",
    ((16, "models/demos/ttnn_resnet/demo/images/"),),
)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_resnet_inference(
    mesh_device, use_program_cache, batch_size, input_loc, imagenet_label_dict, model_location_generator
):
    test_demo_sample(
        mesh_device, use_program_cache, batch_size, input_loc, imagenet_label_dict, model_location_generator
    )
