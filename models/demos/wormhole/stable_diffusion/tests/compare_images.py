import numpy as np
from PIL import Image

from models.utility_functions import comp_pcc


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    return image


def test_pcc():
    # Paths to the images
    path1 = "cpu_image.png"
    path2 = "diffusiondb_0__512x512_ttnn.png"

    # Load the images as numpy arrays
    image1 = load_image(path1)
    image2 = load_image(path2)

    # Compare using PCC
    result, pcc_value = comp_pcc(image1, image2)
    print(f"PCC: {pcc_value:.4f}")
    print(f"Comparison Result: {'Pass' if result else 'Fail'}")
