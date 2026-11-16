"""Dataset discovery, reproducible split creation, and image loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from recyclevision.config import CLASS_NAMES, PROJECT_ROOT, RANDOM_SEED, SPLIT_ROOT
from recyclevision.download_data import ensure_dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_image_paths(dataset_folder: Path) -> tuple[list[str], list[str]]:
    """Collect image paths and matching labels in a stable, sorted order."""
    image_paths: list[str] = []
    image_labels: list[str] = []

    for class_name in CLASS_NAMES:
        class_folder = dataset_folder / class_name
        if not class_folder.is_dir():
            raise FileNotFoundError(f"Expected class folder was not found: {class_folder}")

        # Sorting makes the input order predictable before the seeded data split.
        class_images = sorted(
            path
            for path in class_folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not class_images:
            raise FileNotFoundError(f"No images were found in: {class_folder}")

        image_paths.extend(str(path.resolve()) for path in class_images)
        image_labels.extend([class_name] * len(class_images))

    return image_paths, image_labels


def _save_manifest(paths: list[str], labels: list[str], file_path: Path) -> None:
    """Write file paths and labels to a CSV using project-relative paths."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    relative_paths = [Path(path).resolve().relative_to(PROJECT_ROOT).as_posix() for path in paths]
    manifest = pd.DataFrame({"path": relative_paths, "label": labels})
    manifest.to_csv(file_path, index=False)


def create_or_load_splits(force_recreate: bool = False) -> dict[str, Path]:
    """Create 70/15/15 stratified train/validation/test manifests if needed."""
    split_paths = {
        "train": SPLIT_ROOT / "train.csv",
        "validation": SPLIT_ROOT / "validation.csv",
        "test": SPLIT_ROOT / "test.csv",
    }

    # Reuse the same manifests on repeated training runs unless asked to rebuild them.
    if not force_recreate and all(path.exists() for path in split_paths.values()):
        return split_paths

    dataset_folder = ensure_dataset()
    paths, labels = collect_image_paths(dataset_folder)

    # Reserve 15% for final testing, then 15% of the full dataset for validation.
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    validation_fraction_of_remaining = 0.15 / 0.85
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=validation_fraction_of_remaining,
        random_state=RANDOM_SEED,
        stratify=train_val_labels,
    )

    # Save the paths and labels so training and app test examples use identical splits.
    _save_manifest(train_paths, train_labels, split_paths["train"])
    _save_manifest(validation_paths, validation_labels, split_paths["validation"])
    _save_manifest(test_paths, test_labels, split_paths["test"])

    print(
        "Created stratified splits: "
        f"{len(train_paths)} training, {len(validation_paths)} validation, "
        f"{len(test_paths)} test images."
    )
    return split_paths


class TrashImageDataset(Dataset):
    """Load one image and its integer class label from a saved split CSV."""

    def __init__(self, manifest_path: Path, transform=None) -> None:
        self.rows = pd.read_csv(manifest_path)
        self.transform = transform
        self.label_to_index = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}

    def __len__(self) -> int:
        # DataLoader uses this to determine the number of examples in the split.
        return len(self.rows)

    def __getitem__(self, index: int):
        # Read the manifest row, then turn the stored relative path back into a file path.
        row = self.rows.iloc[index]
        image_path = PROJECT_ROOT / str(row["path"])
        label_name = str(row["label"])

        # Convert all source images to RGB so the pretrained model always gets 3 channels.
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        # The same class order is used for training, metrics, and display labels.
        label = self.label_to_index[label_name]

        # Apply augmentation/normalization after loading the source image.
        if self.transform is not None:
            image = self.transform(image)

        return image, label
