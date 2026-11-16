"""Load a trained checkpoint and classify a single uploaded or test image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from recyclevision.config import CLASS_NAMES, IMAGE_SIZE, MODEL_PATH
from recyclevision.model import build_model, choose_device, make_image_transforms


def load_trained_model(model_path: Path = MODEL_PATH):
    """Load the checkpoint weights and return the model, labels, and device."""
    if not model_path.exists():
        raise FileNotFoundError(
            "The trained checkpoint is missing. Run `python -m recyclevision.train` first."
        )

    device = choose_device()
    # weights_only=True limits checkpoint loading to tensors and simple Python values.
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    class_names = tuple(checkpoint["class_names"])

    # Stop if the checkpoint's label order does not match this version of the app.
    if class_names != CLASS_NAMES:
        raise ValueError("Checkpoint labels do not match the class order in config.py.")

    image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # Return the same resizing and normalization used for the validation images.
    image_transform = make_image_transforms(image_size)["evaluation"]
    return model, class_names, image_transform, device


def predict_image(model, image: Image.Image, image_transform, device: torch.device):
    """Return class probabilities for a PIL image without updating model weights."""
    # Match uploaded color photos to the three-channel images used during training.
    rgb_image = image.convert("RGB")
    input_tensor = image_transform(rgb_image).unsqueeze(0).to(device)

    # inference_mode saves memory and computation because gradients are not needed.
    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    # Move results to ordinary Python-friendly data on the CPU for the Streamlit UI.
    return probabilities.detach().cpu().numpy().astype(np.float64)
