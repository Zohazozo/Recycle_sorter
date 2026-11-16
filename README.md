# Recycle Sorter

Recycle Sorter is an image-classification project that predicts which material appears in a photo: cardboard, glass, metal, paper, plastic, or trash. It has a Streamlit interface for testing uploaded photos and held-out test images.

The project uses transfer learning. It starts with a MobileNetV3-Small image model trained on ImageNet, replaces its final layer with six material categories, and trains the new classifier on the TrashNet dataset. The image model is pretrained; the final recycling categories are learned by this project.

## What you can show in a portfolio

- A complete data pipeline that downloads and unpacks the dataset.
- A reproducible, stratified train/validation/test split.
- Image augmentation and ImageNet normalization.
- Transfer learning with a pretrained convolutional neural network.
- Class-weighted training to account for uneven class counts.
- Validation-based checkpoint selection and a held-out test report.
- An interactive app with image uploads, held-out examples, confidence scores, and a confusion matrix.

## Classes

The classifier predicts one whole-image category:

| Folder label | Display name |
| --- | --- |
| `cardboard` | Cardboard |
| `glass` | Glass |
| `metal` | Metal |
| `paper` | Paper |
| `plastic` | Plastic |
| `trash` | Other trash |

## Requirements

- Python 3.10–3.12
- VS Code with the Microsoft Python extension
- Internet access for the first dataset download and pretrained model download
- About 1 GB of free disk space for the virtual environment, dataset, and model files

Training runs on a CPU if you do not have a supported GPU. It may take several minutes on a laptop. A GPU can make training faster.

## Open and run it in VS Code

### 1. Unzip and open the folder

1. Download and extract `recycle-sorter.zip`.
2. In VS Code, select **File → Open Folder**.
3. Choose the extracted `recycle-sorter` folder.
4. If VS Code asks, install the Microsoft Python extension.
5. Open **Terminal → New Terminal**. The terminal should be in the project folder, where `app.py` and `requirements.txt` are visible.

### 2. Create a virtual environment

Use the command that matches your computer.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the VS Code terminal's Command Prompt profile and run:

```bat
.venv\Scripts\activate.bat
```

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, `(.venv)` should appear at the start of the terminal prompt. In VS Code, choose the interpreter inside this folder with **Python: Select Interpreter** from the Command Palette (`Ctrl+Shift+P` on Windows/Linux or `Cmd+Shift+P` on macOS).

### 3. Install the packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch is a large package. If its installation fails on your computer, use the official [PyTorch Start Locally selector](https://pytorch.org/get-started/locally/) to choose the command for your operating system and CPU/GPU, then install the remaining packages with:

```bash
pip install streamlit pandas scikit-learn Pillow
```

### 4. Train and evaluate the model

```bash
python -m recyclevision.train
```

The first run downloads the TrashNet archive and the pretrained MobileNetV3-Small weights. Training then creates these files:

```text
data/raw/                         downloaded image dataset (kept out of Git)
data/splits/                      train, validation, and test CSV files
artifacts/recycling_model.pth     trained model checkpoint
artifacts/metrics.json            test metrics and training history
artifacts/test_predictions.csv    one row per held-out test image
```

By default, training uses early stopping with a maximum of 8 epochs. You can change the limit:

```bash
python -m recyclevision.train --epochs 12 --batch-size 16
```

Smaller batches use less memory. The first training run may take longer because the dataset and pretrained model need to download.

### 5. Start the app

```bash
streamlit run app.py
```

Open the local URL printed in the terminal. The sidebar lets you upload a JPG, JPEG, or PNG image, or select a labeled image from the held-out test split. The app processes the uploaded image locally and does not save it.

## How the training pipeline works

1. **Download:** `recyclevision/download_data.py` retrieves the resized TrashNet archive from its source repository and safely extracts it under `data/raw/`.
2. **Index and split:** `recyclevision/data.py` checks for the six class folders and creates stratified training, validation, and test CSV files. The fixed random seed makes the split repeatable.
3. **Prepare images:** Training images are resized, randomly flipped, rotated slightly, and normalized. Validation and test images receive resizing and normalization only.
4. **Train:** `recyclevision/model.py` loads MobileNetV3-Small with pretrained ImageNet weights. Its feature extractor is frozen, and the replacement classification layer is trained with AdamW and weighted cross-entropy.
5. **Select and evaluate:** The checkpoint with the lowest validation loss is retained. The test split is evaluated after model selection, and the app shows test accuracy, per-class precision/recall/F1, and a confusion matrix.

## Dataset and limitations

This project uses the [TrashNet dataset](https://github.com/garythung/trashnet), which contains 2,527 images across six classes. The dataset authors describe the photos as individual objects photographed on a white posterboard. That makes it useful for a learning project, but it does not represent cluttered bins, multiple objects, every material type, or real recycling-facility conditions.

The model predicts one broad image category. It does not locate objects in a photo, determine whether a local recycling facility accepts an item, or give authoritative disposal instructions. Check local rules before sorting an actual item. The dataset card lists an MIT license; see the original [dataset repository](https://github.com/garythung/trashnet) for attribution and license details. The pretrained MobileNet weights have their own terms through TorchVision.

## Publish the project on GitHub

First, run training so the model checkpoint and result files exist. The raw dataset is excluded from Git automatically. The trained model is small enough for a normal GitHub repository in most cases; if GitHub warns that it is too large, use Git LFS or leave the checkpoint out and explain the training command in the README.

1. Sign in to GitHub and create a **New repository** named `recycle-sorter`. Choose **Public** if you want employers and recruiters to be able to view it.
2. Keep it empty. Do not initialize it with a README, license, or `.gitignore` because this project already has those files.
3. In the VS Code terminal, make sure you are in the `recycle-sorter` project folder.
4. Replace `YOUR-USERNAME` below with your GitHub username, then run the commands:

```bash
git init
git add .
git status
git commit -m "Build recycling image classifier"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/recycle-sorter.git
git push -u origin main
```

`git status` lets you review the files before committing. Confirm that `data/raw/` is not listed. If Git says your name or email is not configured, set them with the values you use on GitHub:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

On later updates, use:

```bash
git add .
git commit -m "Describe your update"
git push
```

## Project structure

```text
recycle-sorter/
├── app.py                         # Streamlit image testing and metrics dashboard
├── recyclevision/
│   ├── config.py                  # Shared paths, labels, and image settings
│   ├── data.py                    # Dataset discovery, stratified splits, image loading
│   ├── download_data.py           # Dataset download and safe extraction
│   ├── inference.py               # Model loading and single-image prediction
│   ├── model.py                   # MobileNetV3 transfer-learning model
│   └── train.py                   # Training, validation, checkpoint, and test report
├── artifacts/                     # Model and reports generated by training
├── data/                           # Dataset and split files generated by training
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```
