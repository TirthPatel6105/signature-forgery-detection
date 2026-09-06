"""
Signature Forgery Detection — Streamlit App
--------------------------------------------
Loads a trained CNN (signature_model.keras) and offers two modes:

1. Classify — upload one signature image, get a Genuine / Forged prediction.
2. Verify   — upload two signature images, compare them via cosine similarity
              between CNN embeddings (same person vs. different / forged).

Run with:
    streamlit run app.py
"""

import io

import cv2
import numpy as np
import streamlit as st
from numpy.linalg import norm
from PIL import Image
from tensorflow.keras.models import Model, load_model

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = "./signature_model.keras"
IMG_SIZE = 128
CLASSIFY_THRESHOLD = 0.5
SIMILARITY_THRESHOLD = 0.8

st.set_page_config(
    page_title="Signature Forgery Detection",
    page_icon="✍️",
    layout="centered",
)


# --------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_models():
    """Load the trained classifier and derive an embedding sub-model
    (the second-to-last layer's output) for similarity comparisons."""
    classifier = load_model(MODEL_PATH)
    embedding_model = Model(
        inputs=classifier.inputs,
        outputs=classifier.layers[-2].output,
    )
    return classifier, embedding_model


# --------------------------------------------------------------------------
# Preprocessing — mirrors the pipeline used at training time
# --------------------------------------------------------------------------
def preprocess_image(uploaded_file) -> np.ndarray:
    """Convert an uploaded file into a normalized (1, 128, 128, 1) array."""
    image = Image.open(uploaded_file).convert("L")  # grayscale
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = img_array / 255.0
    return img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def classify(model, img_array) -> tuple[str, float]:
    """Return (label, confidence) for a single preprocessed image."""
    pred = model.predict(img_array, verbose=0)[0][0]
    label = "Forged" if pred > CLASSIFY_THRESHOLD else "Genuine"
    confidence = pred if label == "Forged" else 1 - pred
    return label, float(confidence)


def compare(embedding_model, img_array_1, img_array_2) -> tuple[str, float]:
    """Return (verdict, similarity_score) for two preprocessed images."""
    f1 = embedding_model.predict(img_array_1, verbose=0)
    f2 = embedding_model.predict(img_array_2, verbose=0)
    similarity = float(np.dot(f1, f2.T) / (norm(f1) * norm(f2)))
    verdict = "Same person (Genuine)" if similarity > SIMILARITY_THRESHOLD else "Different (Forged)"
    return verdict, similarity


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("✍️ Signature Forgery Detection")
st.caption(
    "A CNN-based system that classifies handwritten signatures as genuine or "
    "forged, and can verify whether two signatures belong to the same person."
)

try:
    classifier_model, embedding_model = get_models()
except (OSError, ValueError):
    st.error(
        f"Couldn't find a trained model at `{MODEL_PATH}`. "
        "Make sure `signature_model.keras` is in the `models/` folder "
        "before running the app."
    )
    st.stop()

mode = st.sidebar.radio("Choose a mode", ["Classify a signature", "Verify two signatures"])
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About**\n\n"
    "Trained on a self-collected dataset of genuine and forged signatures "
    "using a lightweight CNN. See the [README](README.md) for full details "
    "on the pipeline and model comparison."
)

if mode == "Classify a signature":
    st.subheader("Upload a signature to classify")
    uploaded = st.file_uploader("Signature image", type=["png", "jpg", "jpeg"])

    if uploaded:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded, caption="Uploaded signature", use_container_width=True)

        with st.spinner("Analyzing..."):
            img_array = preprocess_image(uploaded)
            label, confidence = classify(classifier_model, img_array)

        with col2:
            if label == "Genuine":
                st.success(f"**{label}**")
            else:
                st.error(f"**{label}**")
            st.metric("Confidence", f"{confidence * 100:.1f}%")

else:
    st.subheader("Upload two signatures to compare")
    col1, col2 = st.columns(2)
    with col1:
        file_a = st.file_uploader("Signature A", type=["png", "jpg", "jpeg"], key="a")
        if file_a:
            st.image(file_a, use_container_width=True)
    with col2:
        file_b = st.file_uploader("Signature B", type=["png", "jpg", "jpeg"], key="b")
        if file_b:
            st.image(file_b, use_container_width=True)

    if file_a and file_b:
        with st.spinner("Comparing..."):
            arr_a = preprocess_image(file_a)
            arr_b = preprocess_image(file_b)
            verdict, similarity = compare(embedding_model, arr_a, arr_b)

        st.markdown("---")
        if "Genuine" in verdict:
            st.success(f"**{verdict}**")
        else:
            st.error(f"**{verdict}**")
        st.metric("Cosine similarity", f"{similarity:.3f}", help=f"Threshold: {SIMILARITY_THRESHOLD}")

st.markdown("---")
st.caption(
    "⚠️ Built on a small, self-collected dataset for a coursework project. "
    "Not benchmarked against public forgery datasets (e.g. CEDAR, GPDS) — "
    "treat predictions as illustrative, not forensic-grade."
)
