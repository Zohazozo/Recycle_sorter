"""Shared project settings used by the downloader, trainer, and app."""

from pathlib import Path


# Resolve paths from this file so commands work from the project root or VS Code.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw"
SPLIT_ROOT = DATA_ROOT / "splits"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"

# These labels match the class-folder names in the TrashNet image archive.
CLASS_NAMES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
DISPLAY_NAMES = {
    "cardboard": "Cardboard",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "trash": "Other trash",
}

# MobileNet's pretrained ImageNet weights expect these input dimensions and values.
IMAGE_SIZE = 160
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
RANDOM_SEED = 42

# Training outputs are kept in a separate folder from source and downloaded photos.
MODEL_PATH = ARTIFACT_ROOT / "recycling_model.pth"
METRICS_PATH = ARTIFACT_ROOT / "metrics.json"
PREDICTIONS_PATH = ARTIFACT_ROOT / "test_predictions.csv"

# The zip contains the resized images and is much smaller than the original archive.
DATASET_URL = (
    "https://github.com/garythung/trashnet/raw/refs/heads/master/"
    "data/dataset-resized.zip"
)
DATASET_ZIP_PATH = RAW_DATA_ROOT / "dataset-resized.zip"
