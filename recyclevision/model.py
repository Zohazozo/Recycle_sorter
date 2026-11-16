"""MobileNetV3 transfer-learning model and image transformations."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models, transforms

from recyclevision.config import CLASS_NAMES, IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD


def choose_device() -> torch.device:
    """Prefer an available accelerator; otherwise train and predict on the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Apple Silicon may support acceleration through PyTorch's Metal backend.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_model(pretrained: bool = True) -> nn.Module:
    """Build MobileNetV3-Small and replace its last layer with six waste classes."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # Freeze the ImageNet feature extractor so this small dataset trains faster.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # The original network predicts ImageNet classes; replace that layer for TrashNet.
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, len(CLASS_NAMES))
    return model


def make_image_transforms(image_size: int = IMAGE_SIZE) -> dict[str, transforms.Compose]:
    """Create separate training and evaluation image preprocessing pipelines."""
    training_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            # Small visual changes help the model handle different object angles.
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06),
            transforms.ToTensor(),
            # Match the input distribution used to train MobileNet's ImageNet weights.
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )

    return {"train": training_transform, "evaluation": evaluation_transform}
