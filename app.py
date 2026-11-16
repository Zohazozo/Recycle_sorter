"""Interactive Streamlit app for exploring the Recycle Sorter model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from recyclevision.config import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    METRICS_PATH,
    MODEL_PATH,
    SPLIT_ROOT,
)
from recyclevision.inference import load_trained_model, predict_image


st.set_page_config(page_title="Recycle Sorter", page_icon="♻️", layout="wide")

# Keep the design clean so the image and model results are the main focus.
st.markdown(
    """
    <style>
      .block-container { max-width: 1180px; padding-top: 2.3rem; }
      [data-testid="stMetric"] {
        background: #f1f7f3;
        border: 1px solid #dcebe0;
        padding: 14px 18px;
        border-radius: 14px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model_and_reports():
    """Load the model once per Streamlit session instead of for every interaction."""
    model, class_names, image_transform, device = load_trained_model()
    with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    return model, class_names, image_transform, device, metrics


st.title("♻️ Recycle Sorter")
st.write(
    "Upload a photo of one item or choose a held-out test image. The model will "
    "estimate its broad material category and show how confident it is."
)

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.warning("Train the model once before opening the image testing app.")
    st.code("python -m recyclevision.train\nstreamlit run app.py", language="bash")
    st.write(
        "The training command downloads the dataset, trains the classifier, and "
        "creates the model checkpoint and evaluation report."
    )
    st.stop()

try:
    with st.spinner("Loading the trained model and evaluation report..."):
        model, class_names, image_transform, device, metrics = get_model_and_reports()
except (OSError, KeyError, ValueError, RuntimeError) as error:
    st.error(f"Could not load the model files: {error}")
    st.stop()

top_metrics = st.columns(3)
top_metrics[0].metric("Held-out test accuracy", f"{metrics['test_accuracy']:.1%}")
top_metrics[1].metric("Material categories", str(len(class_names)))
top_metrics[2].metric("Model", "MobileNetV3-Small")

st.caption(
    "This project classifies one broad category per photo. It is a learning demo, "
    "not a recycling-facility rule checker."
)

test_manifest = SPLIT_ROOT / "test.csv"
input_options = ["Upload a photo"]
if test_manifest.exists():
    input_options.append("Try a held-out test image")

st.sidebar.header("Choose an image")
input_source = st.sidebar.radio("Image source", input_options)
selected_image = None
actual_label = None
image_caption = "Uploaded image"

if input_source == "Upload a photo":
    uploaded_file = st.sidebar.file_uploader(
        "Choose a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        try:
            # Open the upload from memory; this app does not write it to disk.
            selected_image = Image.open(uploaded_file).convert("RGB")
        except OSError:
            st.sidebar.error("That file could not be opened as an image.")
else:
    # The saved test split lets you compare predictions to labels never used for training.
    test_rows = pd.read_csv(test_manifest)
    test_rows["filename"] = test_rows["path"].map(lambda value: Path(value).name)
    selected_row_number = st.sidebar.selectbox(
        "Held-out example",
        options=range(len(test_rows)),
        format_func=lambda row_number: (
            f"{DISPLAY_NAMES[test_rows.iloc[row_number]['label']]}  ·  "
            f"{test_rows.iloc[row_number]['filename']}"
        ),
    )
    selected_row = test_rows.iloc[selected_row_number]
    image_path = Path(__file__).resolve().parent / str(selected_row["path"])
    try:
        selected_image = Image.open(image_path).convert("RGB")
        actual_label = str(selected_row["label"])
        image_caption = f"Held-out test image · actual class: {DISPLAY_NAMES[actual_label]}"
    except OSError:
        st.error("The selected test image could not be opened. Re-run model training to rebuild data files.")

if selected_image is not None:
    try:
        # Convert the chosen photo to class probabilities using the trained model.
        probabilities = predict_image(model, selected_image, image_transform, device)
        predicted_index = int(np.argmax(probabilities))
        predicted_label = class_names[predicted_index]
        predicted_name = DISPLAY_NAMES[predicted_label]
        confidence = float(probabilities[predicted_index])

        image_column, result_column = st.columns([1, 2], gap="large")
        with image_column:
            st.subheader("Image")
            st.image(selected_image, caption=image_caption, use_container_width=True)

        with result_column:
            st.subheader("Model prediction")
            result_columns = st.columns(3)
            result_columns[0].metric("Predicted category", predicted_name)
            result_columns[1].metric("Top score", f"{confidence:.1%}")

            if actual_label is None:
                result_columns[2].metric("Test label", "—")
            else:
                is_correct = predicted_label == actual_label
                result_columns[2].metric(
                    "Test label",
                    DISPLAY_NAMES[actual_label],
                    delta="Correct" if is_correct else "Different",
                    delta_color="normal" if is_correct else "inverse",
                )

            # Show the model's scores for every category, not only the top guess.
            probability_table = pd.DataFrame(
                {
                    "Category": [DISPLAY_NAMES[name] for name in class_names],
                    "Probability": probabilities,
                }
            ).sort_values("Probability", ascending=False)
            st.write("**Scores for all categories**")
            st.bar_chart(probability_table, x="Category", y="Probability")

        if confidence < 0.55:
            st.info(
                "The model is uncertain about this photo. A clear picture of one "
                "item against a simple background may be easier to classify."
            )
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"The image could not be classified: {error}")
else:
    st.info("Choose an input in the sidebar to see a prediction.")

st.divider()
st.header("How the model performed")
st.write(
    "These results come from a held-out test split. The confusion matrix shows "
    "which categories the model mixes up most often."
)

report = metrics["classification_report"]
class_rows = []
for class_name in CLASS_NAMES:
    class_metrics = report[class_name]
    class_rows.append(
        {
            "Category": DISPLAY_NAMES[class_name],
            "Precision": class_metrics["precision"],
            "Recall": class_metrics["recall"],
            "F1 score": class_metrics["f1-score"],
            "Test images": int(class_metrics["support"]),
        }
    )

report_column, matrix_column = st.columns([1, 1], gap="large")
with report_column:
    st.subheader("Per-category scores")
    st.dataframe(pd.DataFrame(class_rows), hide_index=True, use_container_width=True)

with matrix_column:
    st.subheader("Confusion matrix")
    confusion_matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[DISPLAY_NAMES[name] for name in class_names],
        columns=[DISPLAY_NAMES[name] for name in class_names],
    )
    confusion_matrix.index.name = "Actual"
    confusion_matrix.columns.name = "Predicted"
    st.dataframe(confusion_matrix, use_container_width=True)

st.caption(
    "A high confidence score does not guarantee a correct prediction. Local "
    "recycling rules vary, so check your local facility's guidance before sorting."
)
