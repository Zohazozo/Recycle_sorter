"""Download and unpack the resized TrashNet dataset."""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from recyclevision.config import DATASET_URL, DATASET_ZIP_PATH, RAW_DATA_ROOT


def _is_dataset_folder(folder: Path) -> bool:
    """Check whether a folder directly contains all six class subfolders."""
    if not folder.is_dir():
        return False

    child_folder_names = {child.name.lower() for child in folder.iterdir() if child.is_dir()}
    expected_names = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
    return expected_names.issubset(child_folder_names)


def find_dataset_folder(search_root: Path = RAW_DATA_ROOT) -> Path | None:
    """Find the folder with class subfolders, even if the zip adds a wrapper folder."""
    if not search_root.exists():
        return None

    # Check the root first because some copies of the archive have no wrapper folder.
    if _is_dataset_folder(search_root):
        return search_root

    # Search nested directories because archive layouts can add one or more prefixes.
    for candidate in search_root.rglob("*"):
        if _is_dataset_folder(candidate):
            return candidate

    return None


def _download_archive(destination: Path) -> None:
    """Stream the zip file to disk so the full archive is not held in memory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "recycle-sorter-portfolio-project"},
    )

    print("Downloading the resized TrashNet image archive...")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with destination.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)
    except (OSError, urllib.error.URLError) as error:
        # Remove a partial file so the next run starts with a clean download.
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            "The dataset download failed. Check your internet connection and try again."
        ) from error

    print(f"Downloaded {destination.stat().st_size / (1024 * 1024):.1f} MB.")


def _extract_archive(archive_path: Path, destination: Path) -> None:
    """Extract the dataset after checking that zip members stay inside the folder."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()

    try:
        with zipfile.ZipFile(archive_path) as archive:
            # Reject paths such as ../../outside.txt before extracting any files.
            for member in archive.infolist():
                resolved_target = (destination / member.filename).resolve()
                if not resolved_target.is_relative_to(destination_resolved):
                    raise RuntimeError("The downloaded zip contains an unsafe file path.")

            archive.extractall(destination)
    except zipfile.BadZipFile as error:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(
            "The downloaded file was not a valid zip. Run the command again to retry."
        ) from error


def ensure_dataset() -> Path:
    """Return the extracted dataset folder, downloading it if it is not present."""
    existing_folder = find_dataset_folder()
    if existing_folder is not None:
        print(f"Using dataset already on disk: {existing_folder}")
        return existing_folder

    if not DATASET_ZIP_PATH.exists():
        _download_archive(DATASET_ZIP_PATH)

    _extract_archive(DATASET_ZIP_PATH, RAW_DATA_ROOT)
    dataset_folder = find_dataset_folder()
    if dataset_folder is None:
        raise RuntimeError(
            "The archive was extracted, but the six expected class folders were not found. "
            "See the TrashNet link in README.md for the dataset source."
        )

    print(f"Dataset is ready: {dataset_folder}")
    return dataset_folder


if __name__ == "__main__":
    # This entry point lets you download data separately before training.
    ensure_dataset()
