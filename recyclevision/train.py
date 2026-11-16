"""Train, validate, and test the recycling image classifier."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from recyclevision.config import (
    ARTIFACT_ROOT,
    CLASS_NAMES,
    IMAGE_SIZE,
    METRICS_PATH,
    MODEL_PATH,
    PREDICTIONS_PATH,
    RANDOM_SEED,
)
from recyclevision.data import TrashImageDataset, create_or_load_splits
from recyclevision.model import build_model, choose_device, make_image_transforms


def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    """Fix Python, NumPy, and PyTorch random seeds for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_data_loaders(batch_size: int, workers: int = 0):
    """Create shuffled training batches and stable validation/test data loaders."""
    split_paths = create_or_load_splits()
    image_transforms = make_image_transforms(IMAGE_SIZE)

    datasets = {
        "train": TrashImageDataset(split_paths["train"], image_transforms["train"]),
        "validation": TrashImageDataset(
            split_paths["validation"], image_transforms["evaluation"]
        ),
        "test": TrashImageDataset(split_paths["test"], image_transforms["evaluation"]),
    }

    # num_workers=0 is the most reliable setting for beginners using Windows.
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
        ),
    }
    return datasets, loaders


def make_class_weights(training_manifest: Path, device: torch.device) -> torch.Tensor:
    """Give less common categories more influence in the cross-entropy loss."""
    labels = pd.read_csv(training_manifest)["label"].tolist()
    counts = np.array([labels.count(class_name) for class_name in CLASS_NAMES], dtype=np.float32)

    # The inverse-frequency formula balances each class without oversampling files.
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_one_epoch(model, loader, loss_function, device, optimizer=None):
    """Run one training or validation pass and return average loss and accuracy."""
    training_mode = optimizer is not None
    model.train(training_mode)

    if training_mode:
        # Keep the frozen ImageNet backbone's BatchNorm statistics unchanged.
        # The new classifier remains in training mode and continues to learn.
        model.features.eval()
        model.classifier.train()

    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training_mode:
            # Clear old gradients so this batch's update uses only its own error.
            optimizer.zero_grad(set_to_none=True)

        # Only build the gradient graph during training to save memory in validation.
        with torch.set_grad_enabled(training_mode):
            logits = model(images)
            loss = loss_function(logits, labels)

            if training_mode:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        correct_predictions += int((predictions == labels).sum().item())
        total_examples += batch_size

    average_loss = total_loss / max(total_examples, 1)
    accuracy = correct_predictions / max(total_examples, 1)
    return average_loss, accuracy


def predict_loader(model, loader, device):
    """Collect test labels, predicted labels, and top scores in dataset order."""
    model.eval()
    actual_labels: list[int] = []
    predicted_labels: list[int] = []
    confidence_scores: list[float] = []

    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)
            scores, predictions = probabilities.max(dim=1)

            actual_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.cpu().tolist())
            confidence_scores.extend(scores.cpu().tolist())

    return actual_labels, predicted_labels, confidence_scores


def _write_training_outputs(
    model,
    device,
    history: list[dict[str, float]],
    test_loader,
    test_manifest: Path,
    training_settings: dict[str, int | float],
) -> None:
    """Save the best model, a final held-out report, and per-image predictions."""
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    # The checkpoint holds only tensors and simple metadata so it can load safely.
    checkpoint = {
        "state_dict": model.state_dict(),
        "class_names": list(CLASS_NAMES),
        "image_size": IMAGE_SIZE,
        "architecture": "mobilenet_v3_small",
    }
    torch.save(checkpoint, MODEL_PATH)

    # Evaluate the untouched test set only after the best validation checkpoint is chosen.
    actual, predicted, confidence = predict_loader(model, test_loader, device)
    per_class_report = classification_report(
        actual,
        predicted,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(
        actual,
        predicted,
        labels=list(range(len(CLASS_NAMES))),
    )

    # Convert NumPy scalar values to standard Python types for JSON serialization.
    metrics = {
        "test_accuracy": float(accuracy_score(actual, predicted)),
        "class_names": list(CLASS_NAMES),
        "classification_report": per_class_report,
        "confusion_matrix": confusion.tolist(),
        "training_history": history,
        "training_settings": training_settings,
    }
    METRICS_PATH.write_text(
        json.dumps(_to_json_compatible(metrics), indent=2), encoding="utf-8"
    )

    # Keep a row-level file so the held-out results can be inspected outside the app.
    test_rows = pd.read_csv(test_manifest)
    prediction_table = test_rows.copy()
    prediction_table["predicted_label"] = [CLASS_NAMES[index] for index in predicted]
    prediction_table["confidence"] = confidence
    prediction_table.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Held-out test accuracy: {metrics['test_accuracy']:.1%}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


def _to_json_compatible(value):
    """Convert NumPy values inside metric reports to standard JSON types."""
    if isinstance(value, dict):
        return {key: _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def train_model(epochs: int = 8, batch_size: int = 32, patience: int = 3) -> None:
    """Train the replacement classification head and keep the best validation model."""
    set_random_seeds()
    _, loaders = make_data_loaders(batch_size=batch_size)
    split_paths = create_or_load_splits()
    device = choose_device()
    print(f"Using device: {device}")

    # ImageNet weights provide useful visual features before training on TrashNet.
    model = build_model(pretrained=True).to(device)
    class_weights = make_class_weights(split_paths["train"], device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)

    # Only the newly replaced final layer is trainable in this transfer-learning setup.
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=0.001, weight_decay=0.0001)

    best_validation_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    started_at = time.time()

    for epoch in range(1, epochs + 1):
        # The optimizer is passed only for training; validation runs without updates.
        train_loss, train_accuracy = run_one_epoch(
            model, loaders["train"], loss_function, device, optimizer
        )
        validation_loss, validation_accuracy = run_one_epoch(
            model, loaders["validation"], loss_function, device
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_loss:.4f}, train accuracy {train_accuracy:.1%} | "
            f"validation loss {validation_loss:.4f}, "
            f"validation accuracy {validation_accuracy:.1%}"
        )

        # Validation loss chooses the checkpoint; the test data is not used here.
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Stop early if validation loss has not improved for several epochs.
        if epochs_without_improvement >= patience:
            print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
            break

    model.load_state_dict(best_state)
    elapsed_seconds = time.time() - started_at
    print(f"Training finished in {elapsed_seconds / 60:.1f} minutes.")

    _write_training_outputs(
        model=model,
        device=device,
        history=history,
        test_loader=loaders["test"],
        test_manifest=split_paths["test"],
        training_settings={
            "epochs_requested": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "best_epoch": best_epoch,
            "image_size": IMAGE_SIZE,
        },
    )


def parse_args() -> argparse.Namespace:
    """Read optional training settings from the terminal command."""
    parser = argparse.ArgumentParser(description="Train the Recycle Sorter classifier.")
    parser.add_argument("--epochs", type=int, default=8, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Images per update.")
    parser.add_argument("--patience", type=int, default=3, help="Early-stopping patience.")
    return parser.parse_args()


if __name__ == "__main__":
    # `python -m recyclevision.train` enters the training pipeline here.
    arguments = parse_args()
    train_model(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        patience=arguments.patience,
    )
