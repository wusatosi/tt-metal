# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import time

import torch
from torch import nn
from loguru import logger

import torchvision.transforms as transforms
from torchvision.datasets import MNIST as mnist_dataset
from torch.utils.data import DataLoader


# Batch size configurations
MNIST_BATCH_SIZE_EXP_RANGE = 7

# Input size configurations
MNIIST_INPUT_SIZE_EXP_RANGE = [5, 7]
MNIIST_INPUT_SIZE_FACTORS = [1, 3, 5, 7]

# Hidden layer size configurations
MNIST_HIDDEN_SIZE_EXP_RANGE = [5, 7]
MNIIST_HIDDEN_SIZE_FACTORS = [1, 3]

MNIST_INPUT_FEATURE_SIZE = 784  # 784 = 28 * 28, default size of MNIST image
MNIST_OUTPUT_FEATURE_SIZE = 10  # 10 classes in MNIST, default output size
MNIIST_HIDDEN_SIZE = 256  # Hidden layer size, default size


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


def load_dataset(batch_size, dtype=torch.float32):
    """
    Load and normalize MNIST dataset
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Mean and std for MNIST
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten image
            transforms.Lambda(lambda x: x.to(dtype)),  # Convert to dtype
        ]
    )

    train_dataset = mnist_dataset(root="./data", train=True, download=True, transform=transform)
    test_dataset = mnist_dataset(root="./data", train=False, download=True, transform=transform)

    # Shuffle training data so that shuffling is not done in the training loop
    # This is to ensure that the same data is used for both Torch and Forge
    indices = torch.randperm(len(train_dataset))
    train_dataset = [train_dataset[i] for i in indices]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return test_loader, train_loader


@pytest.mark.push
def test_mnist_training():
    torch.manual_seed(0)

    # Model and data type.
    # For bfloat16, the following line should be added to the test_forge_vs_torch function:
    # In file forge/forge/op/eval/forge/eltwise_unary.py:418 should be replaced with: threshold_tensor = ac.tensor(torch.zeros(shape, dtype=torch.bfloat16) + threshold)
    # That sets relu threshold to bfloat16 tensor.
    # And in file forge/forge/compile.py::compile_main forced bfloat 16 should be added compiler_cfg.default_df_override = DataFormat.Float16_b
    dtype = torch.float32

    # Set training hyperparameters
    num_epochs = 20
    batch_size = 2048
    learning_rate = 0.1

    # Load dataset
    test_loader, train_loader = load_dataset(batch_size, dtype=dtype)

    # Define model and instruct it to compile and run on TT device
    framework_model = MNISTLinear()  # bias=False because batch_size=1 with bias=True is not supported

    # Create a torch loss and leave on CPU
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define optimizer and instruct it to compile and run on TT device
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=learning_rate)

    logger.info("Starting training loop... (logger will be disabled)")
    logger.disable("")
    for epoch_idx in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Reset gradients (every batch)
            framework_optimizer.zero_grad()

            pred = framework_model(data)

            # Create target tensor and leave on CPU
            target = nn.functional.one_hot(target, num_classes=10).to(dtype)

            # Forward pass (prediction) on device

            golden_pred = framework_model(data)
            assert golden_pred.dtype == dtype

            # Compute loss on CPU
            loss = loss_fn(pred, target)
            total_loss += loss.item()

            golden_loss = loss_fn(golden_pred, target)
            assert torch.allclose(loss, golden_loss, rtol=1e-1)  # 10% tolerance

            # Run backward pass on device
            loss.backward()

            # Adjust weights (on CPU)
            framework_optimizer.step()

        print(f"epoch: {epoch_idx} loss: {total_loss}")

    test_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        pred = framework_model(data)
        target = nn.functional.one_hot(target, num_classes=10).to(dtype)

        test_loss += loss_fn(pred, target)

    print(f"Test (total) loss: {test_loss}")

    # Set model to evaluation mode
    framework_model.eval()

    # Initialize variables to track correct predictions and total samples
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for data, target in test_loader:
            # Forward pass: Get predictions
            pred = framework_model(data)

            # Convert predictions to class indices
            predicted_classes = torch.argmax(pred, dim=1)

            # Count correct predictions
            correct += (predicted_classes == target).sum().item()
            total += target.size(0)

    # Compute accuracy
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    torch.save(framework_model.state_dict(), "./model.pth")


test_mnist_training()
