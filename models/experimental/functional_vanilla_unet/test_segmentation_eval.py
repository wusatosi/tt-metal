import pytest
import os
import torch
import ttnn
import argparse
from tqdm import tqdm
from skimage.io import imsave
import numpy as np
from models.experimental.functional_vanilla_unet.demo import demo_utils


def iou(y_true, y_pred):
    """Computes Intersection over Union (IoU)."""
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / union if union != 0 else 0


def dice_score(y_true, y_pred):
    """Computes Dice Score (F1 Score for segmentation)."""
    intersection = np.logical_and(y_true, y_pred).sum()
    return (2 * intersection) / (y_true.sum() + y_pred.sum()) if (y_true.sum() + y_pred.sum()) != 0 else 0


def pixel_accuracy(y_true, y_pred):
    """Computes Pixel Accuracy."""
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    """Computes Precision (Positive Predictive Value)."""
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    return tp / (tp + fp) if (tp + fp) != 0 else 0


def recall(y_true, y_pred):
    """Computes Recall (Sensitivity)."""
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    return tp / (tp + fn) if (tp + fn) != 0 else 0


def f1_score(y_true, y_pred):
    """Computes F1 Score (Harmonic Mean of Precision and Recall)."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (2 * p * r) / (p + r) if (p + r) != 0 else 0


def evaluation(device, res, model_type, model, input_dtype, input_memory_config=None, model_name=None):
    args = argparse.Namespace(
        device="cpu",  # Choose "cpu" or "cuda:0" based on your setup
        batch_size=1,
        weights="models/experimental/functional_vanilla_unet/unet.pt",  # Path to the pre-trained model weights
        images="models/experimental/functional_vanilla_unet/demo/imageset",  # Path to your input image
        image_size=(480, 640),  # Resize input image to this size
        predictions="models/experimental/functional_vanilla_unet/demo/pred_image_set",  # Directory to save prediction results
    )
    loader = demo_utils.data_loader_imageset(args)

    input_list = []
    pred_list = []
    true_list = []
    metrics_list = []

    for i, data in tqdm(enumerate(loader)):
        x, y_true = data

        if model_type == "torch_model":
            y_pred = model(x)
        else:
            ttnn_input_tensor = ttnn.from_torch(
                x.permute(0, 2, 3, 1), device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            y_pred = model(device, ttnn_input_tensor, False)
        y_pred_np = y_pred.detach().cpu().numpy()
        pred_list.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])

        y_true_np = y_true.detach().cpu().numpy()
        true_list.extend([y_true_np[s] for s in range(y_true_np.shape[0])])

        x_np = x.detach().cpu().numpy()
        input_list.extend([x_np[s] for s in range(x_np.shape[0])])

    volumes = demo_utils.postprocess_per_volume(
        input_list,
        pred_list,
        true_list,
        loader.dataset.patient_slice_index,
        loader.dataset.patients,
    )

    metrics_list = []
    for p in volumes:
        x = volumes[p][0]
        y_pred = volumes[p][1]
        y_true = volumes[p][2]
        y_true = (y_true == 255).astype(np.uint8)  # Convert 255 → 1
        y_pred = y_pred.astype(np.uint8)

        metrics = {
            "IoU": iou(y_true, y_pred),
            "Dice Score": dice_score(y_true, y_pred),
            "Pixel Accuracy": pixel_accuracy(y_true, y_pred),
            "Precision": precision(y_true, y_pred),
            "Recall": recall(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
        }

        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        for s in range(x.shape[0]):
            image = demo_utils.gray2rgb(x[s, 1])  # channel 1 is for FLAIR
            image = demo_utils.outline(image, y_pred[s, 0], color=[255, 0, 0])
            image = demo_utils.outline(image, y_true[s, 0], color=[0, 255, 0])
            filename = "{}-{}.png".format(p, str(s).zfill(2))
            filepath = os.path.join(args.predictions, filename)
            imsave(filepath, image)


@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("res", [(320, 320)])
def test_vanilla_unet(device, model_type, res, model_location_generator, reset_seeds):
    from models.experimental.functional_vanilla_unet.reference.unet import UNet
    from models.experimental.functional_vanilla_unet.ttnn.ttnn_unet import TtUnet
    from models.experimental.functional_vanilla_unet.ttnn.model_preprocesser import create_custom_preprocessor
    from ttnn.model_preprocessing import preprocess_model_parameters

    weights_path = "models/experimental/functional_vanilla_unet/unet.pt"
    if not os.path.exists(weights_path):
        os.system("bash models/experimental/functional_vanilla_unet/weights_download.sh")

    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    ds_state_dict = {k: v for k, v in state_dict.items()}

    reference_model = UNet()

    new_state_dict = {}
    keys = [name for name, parameter in reference_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )
    ttnn_model = TtUnet(device=device, parameters=parameters, model=reference_model)

    if not os.path.exists("models/experimental/functional_vanilla_unet/demo/imageset"):
        os.system("python models/experimental/functional_vanilla_unet/dataset_download.py")

    model_name = "vanilla_unet"
    input_dtype = ttnn.bfloat16
    input_memory_config = ttnn.L1_MEMORY_CONFIG

    evaluation(
        device=device,
        res=res,
        model_type=model_type,
        model=ttnn_model if model_type == "tt_model" else reference_model,
        input_dtype=input_dtype,
        input_memory_config=input_memory_config,
        model_name=model_name,
    )
