# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from PIL import Image
import torch
import os
import glob
from models.sample_data.huggingface_imagenet_classes import IMAGENET2012_CLASSES
from datasets import load_dataset
from transformers import AutoFeatureExtractor
from torchvision import datasets, transforms


def get_data_loader_cifar10(batch_size, iterations):

    # 1. Load feature extractor for preprocessing
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/deit-tiny-patch16-224")

    # 2. Define transform: resize to 224x224, normalize as expected by DeiT
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ])

    cifar10_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    cifar10_loader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=False)

    def loader():
        iteration_count = 0
        for images, labels in cifar10_loader:
            if iteration_count >= iterations:
                break
            examples = []
            for img, label in zip(images, labels):
                examples.append(InputExample(image=img, label=label.item()))
            yield examples
            iteration_count += 1

    return loader()

def get_cifar10_label_dict():
    """
    Returns a dictionary mapping CIFAR-10 class indices to class names.
    """
    return {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }

def get_batch_cifar(data_loader):
    loaded_images = next(data_loader)
    images = []
    labels = []
    for image in loaded_images:
        images.append(image.image.unsqueeze(0))  # Add batch dimension to each image
        labels.append(image.label)
    
    images = torch.cat(images, dim=0)  # Concatenate all images into a single tensor
    return images, labels

class InputExample(object):
    def __init__(self, image, label=None):
        self.image = image
        self.label = label


def get_input(image_path):
    img = Image.open(image_path)
    return img


def get_label(image_path):
    _, image_name = image_path.rsplit("/", 1)
    image_name_exact, _ = image_name.rsplit(".", 1)
    _, label_id = image_name_exact.rsplit("_", 1)
    label = list(IMAGENET2012_CLASSES).index(label_id)
    return label


def get_data_loader(input_loc, batch_size, iterations):
    img_dir = input_loc + "/"
    data_path = os.path.join(img_dir, "*G")
    files = glob.glob(data_path)

    def loader():
        examples = []
        for f1 in files:
            examples.append(
                InputExample(
                    image=get_input(f1),
                    label=get_label(f1),
                )
            )
            if len(examples) == batch_size:
                yield examples
                del examples
                examples = []

    def loader_hf():
        examples = []
        for f1 in files:
            examples.append(
                InputExample(
                    image=f1["image"],
                    label=f1["label"],
                )
            )
            if len(examples) == batch_size:
                yield examples
                del examples
                examples = []

    if len(files) == 0:
        files_raw = iter(load_dataset("imagenet-1k", split="validation", use_auth_token=True, streaming=True))
        files = []
        sample_count = batch_size * iterations
        for _ in range(sample_count):
            files.append(next(files_raw))
        del files_raw
        return loader_hf()

    return loader()


def get_batch(data_loader, image_processor):
    loaded_images = next(data_loader)
    images = None
    labels = []
    for image in loaded_images:
        img = image.image
        labels.append(image.label)
        if img.mode == "L":
            img = img.convert(mode="RGB")
        img = image_processor(img, return_tensors="pt")
        img = img["pixel_values"]

        if images is None:
            images = img
        else:
            images = torch.cat((images, img), dim=0)
    return images, labels


def get_data(input_loc):
    img_dir = input_loc + "/"
    data_path = os.path.join(img_dir, "*G")
    files = sorted(glob.glob(data_path))
    examples = []
    for f1 in files:
        examples.append(
            InputExample(
                image=get_input(f1),
                label=get_label(f1),
            )
        )
    image_examples = examples

    return image_examples
