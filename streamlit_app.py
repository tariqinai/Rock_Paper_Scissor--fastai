# streamlit_app.py
import os, pathlib, io
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import pandas as pd
import torch

# --- WindowsPath -> PosixPath patch for Windows-exported pickles on Linux ---
if os.name != "nt":  # running on Linux/Mac
    pathlib.WindowsPath = pathlib.PosixPath
# ---------------------------------------------------------------------------

# Import fastai AFTER the patch so unpickling works
from fastai.vision.all import load_learner, PILImage

# ---------- App Config ----------
st.set_page_config(
    page_title="Rock • Paper • Scissors - fastai",
    page_icon="✋",
    layout="centered"
)

# ---------- Helpers ----------
def _to_rgb_contiguous(img: Image.Image) -> Image.Image:
    """Normalize EXIF orientation, remove alpha, ensure contiguous RGB uint8."""
    img = ImageOps.exif_transpose(img)  # handle rotated images

    if img.mode == "P":         # paletted PNGs
        img = img.convert("RGBA")
    if img.mode == "RGBA":      # composite onto white background
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.array(img, dtype=np.uint8, copy=True)  # copy=True -> contiguous
    return Image.fromarray(arr, mode="RGB")

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load the fastai exported Learner (.pkl). Cached so it loads once."""
    # keep CPU-friendly and deterministic
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    learn = load_learner(model_path, cpu=True)

    # best-effort: get classes for a nice probability table
    try:
        classes = list(learn.dls.vocab)
    except Exception:
        classes = None

    # prediction-time safety: avoid multiprocessing workers
    try:
        learn.dls.num_workers = 0
    except Exception:
        pass

    return learn, classes

def predict_image(learn, img: Image.Image) -> Tuple[str, float, List[float]]:
    """Run prediction on a PIL image and return (label, confidence, probs_list)."""
    img = _to_rgb_contiguous(img)
    with torch.inference_mode():
        pred_class, pred_idx, probs = learn.predict(PILImage.create(img))
    confidence = float(probs[pred_idx])
    return str(pred_class), confidence, [float(p) for p in probs]

# ---------- Sidebar ----------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Select page", ["Single Image", "Batch (multiple images)"])

st.sidebar.markdown("---")
st.sidebar.caption("Model: fastai exported .pkl")

# Path input (default to a file named rps.pkl next to this script)
default_model_path = Path(__file__).parent / "rps.pkl"
model_file = st.sidebar.text_input("Path to .pkl", value=str(default_model_path))

if not Path(model_file).exists():
    st.sidebar.warning("Model file not found. Upload 'rps.pkl' next to this script or enter a valid path.")
    st.stop()
else:
    st.sidebar.success("Model found ✅")

# ---------- Load the model ----------
try:
    with st.spinner("Loading model..."):
        learn, classes = load_model(model_file)
except Exception as e:
    st.error("❌ Failed to load model. See details below.")
    st.exception(e)
    st.stop()

# ---------- UI ----------
st.title("✊ ✋ ✌️ Rock • Paper • Scissors")
st.caption("Fastai model inference • Upload images and get predictions")

# ---------- Single Image Page ----------
if page == "Single Image":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if file is not None:
        try:
            img = Image.open(file)
            st.image(img, caption="Uploaded image", use_container_width=True)

            with st.spinner("Predicting..."):
                label, conf, probs = predict_image(learn, img)

            st.subheader(f"Prediction: **{label}**")
            st.write(f"Confidence: **{conf*100:.2f}%**")

            if classes is not None:
                prob_table = pd.DataFrame({
                    "class": classes,
                    "probability": probs,
                    "percent": [round(p*100, 2) for p in probs],
                }).sort_values("probability", ascending=False)
                st.dataframe(prob_table, use_container_width=True)

        except Exception as e:
            st.error("Prediction failed. See details below.")
            st.exception(e)

# ---------- Batch Page ----------
else:
    files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True
    )
    if files:
        rows = []
        for f in files:
            try:
                img = Image.open(f)
                label, conf, probs = predict_image(learn, img)
                rows.append({
                    "filename": f.name,
                    "prediction": label,
                    "confidence (%)": round(conf*100, 2)
                })
            except Exception as e:
                rows.append({
                    "filename": f.name,
                    "prediction": f"Error: {e}",
                    "confidence (%)": ""
                })

        df = pd.DataFrame(rows)
        st.success(f"Predicted {len(rows)} image(s)")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇️ Download results as CSV",
            df.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv"
        )

st.markdown("---")
st.caption("Built with fastai + Streamlit. Place `rps.pkl` alongside this file or set the path in the sidebar.")
