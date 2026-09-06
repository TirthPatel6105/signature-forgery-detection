# ✍️ Signature Forgery Detection

A CNN-based system that classifies handwritten signatures as **genuine or forged**, and can also **verify** whether two signatures belong to the same person by comparing learned embeddings.

Built as a self-directed deep learning project — including collecting and preparing the dataset from scratch.

---

## Overview

Signature verification is a classic biometric problem: given a signature, decide whether it's authentic or a forgery. This project tackles it two ways:

1. **Classification** — a CNN predicts genuine vs. forged directly from a single signature image.
2. **Verification** — embeddings extracted from the trained CNN are compared via cosine similarity to judge whether two signatures were made by the same person.

Three CNN architectures of increasing depth were built and benchmarked to find the best fit for the dataset size.

---

## Dataset

The dataset was **self-collected**, not sourced from a public benchmark:

- Signature sheets were scanned as PDFs, each laid out as a 3×4 grid (12 signatures per page), for both genuine and forged sets.
- PDFs were converted to high-resolution PNGs (300 DPI) and each grid cell was programmatically cropped using OpenCV — removing table gridlines, thresholding to locate ink pixels, and tightly cropping around the signature.
- Final dataset: **552 images** (276 genuine + 276 forged), resized to 128×128 grayscale and normalized to [0, 1].

**Raw signature images are intentionally excluded from this repository** to protect the privacy of the individuals who contributed samples. Only code, pipeline, and results are shared.

> This is a small, self-collected dataset — not benchmarked against public forgery datasets (e.g. CEDAR, GPDS). Results should be read as a coursework-scale proof of concept, not a production-grade forensic tool.

---

## Pipeline

1. **PDF → image**: `pdf2image` at 300 DPI.
2. **Grid cropping**: each page split into a 3×4 grid; borders trimmed to remove table lines.
3. **Signature localization**: binary thresholding (`cv2.threshold`) to find ink pixels, then a tight bounding-box crop with padding.
4. **Preprocessing**: grayscale, resize to 128×128, normalize to [0, 1].
5. **Split**: 80/20 stratified train/test split, shuffled (`random_state=42`).
6. **Augmentation**: light rotation (±5°), zoom (5%), and shift (3%) — kept deliberately subtle, since aggressive augmentation would distort the exact stroke shapes that distinguish genuine from forged signatures.

---

## Model comparison

Three CNN architectures were trained and compared:

| Model | Architecture | Test Accuracy |
|---|---|---|
| **Model 1** | 2 conv layers (32→64 filters), no batch norm/dropout | **84.7%** ✅ best |
| Model 2 | + BatchNorm, larger dense layer (128), Dropout(0.5) | 82.9% |
| Model 3 | 4 conv blocks (32→64→128→256), BatchNorm throughout | Unstable, underperformed |

**Key finding:** the simplest model performed best. With only ~440 training images, the deeper architectures had far more parameters than the data could support, leading to overfitting and unstable training rather than better generalization — a practical illustration of the model-capacity-vs-dataset-size trade-off.

Training used `ReduceLROnPlateau` (monitoring validation loss) to lower the learning rate on plateaus.

---

## Signature verification mode

Beyond classification, embeddings are extracted from the trained model's second-to-last layer and compared between two signatures using **cosine similarity** — conceptually a lightweight version of what Siamese networks do for one-shot verification. A similarity above `0.8` is treated as "same person."

---

## Repo structure

```
.
├── app.py                  # Streamlit app (classify + verify modes)
├── notebooks/
│   └── signature_forgery.ipynb   # Full pipeline: data prep, training, evaluation
├── signature_model.keras     # Trained model (not committed if large — see below)
├── requirements.txt
├── .gitignore
└── README.md
```

> The raw dataset folders and the trained model file are excluded via `.gitignore`. The model can be shared on request or via a release/artifact link rather than committed directly.

---

## Setup

```bash
git clone https://github.com/TirthPatel6105/signature-forgery-detection.git
cd signature-forgery-detection
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

This launches a local web app with two modes:
- **Classify a signature** — upload one image, get a Genuine/Forged prediction with a confidence score.
- **Verify two signatures** — upload two images, get a similarity score and same-person/different verdict.

---

## Limitations

- Small, self-collected dataset (552 images) — not benchmarked against standard forgery datasets.
- No separate validation set was held out from the test set during model selection; with more data, a proper train/val/test split or k-fold cross-validation would give a less biased performance estimate.
- Verification mode uses embeddings from a classifier, not a purpose-trained Siamese/triplet network — a dedicated verification architecture would likely perform better.

## Possible improvements

- Collect a larger, more diverse dataset (more signers, more forgery styles).
- Train a dedicated Siamese network for verification instead of reusing classifier embeddings.
- Evaluate on a public benchmark (CEDAR/GPDS) for a more rigorous accuracy comparison.

---

## Tech stack

Python · TensorFlow/Keras · OpenCV · scikit-learn · Streamlit
