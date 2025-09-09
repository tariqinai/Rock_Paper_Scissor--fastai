import io
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from PIL import Image
import pandas as pd

# fastai imports
from fastai.vision.all import load_learner, PILImage
import torch

# ---------- App Config ----------
st.set_page_config(page_title="Rock • Paper • Scissors - fastai",
                   page_icon="✋", layout="centered")

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    "Load the fastai exported Learner (.pkl). Cached so it loads once."
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    learn = load_learner(model_path)
    try:
        classes = list(learn.dls.vocab)
    except Exception:
        classes = None
    return learn, classes

def predict_image(learn, img: Image.Image) -> Tuple[str, float, List[float]]:
    "Run prediction on a PIL image and return (label, confidence, probs_list)."
    if img.mode != "RGB":
        img = img.convert("RGB")
    pred_class, pred_idx, probs = learn.predict(PILImage.create(img))
    confidence = float(probs[pred_idx])
    return str(pred_class), confidence, [float(p) for p in probs]

# ---------- Sidebar Navigation ----------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Select page", ["Single Image", "Batch (multiple images)"])

st.sidebar.markdown("---")
st.sidebar.caption("Model: fastai exported .pkl")

# ---------- Load model (.pkl) ----------
default_model_path = Path("rps.pkl")
model_file = st.sidebar.text_input("Path to .pkl", value=str(default_model_path))
model_exists = Path(model_file).exists()

if not model_exists:
    st.sidebar.warning("Model file not found. Upload 'rps.pkl' to the app directory (HF Space) or enter a valid path.")
    st.stop()
else:
    st.sidebar.success("Model found ✅")

with st.spinner("Loading model..."):
    learn, classes = load_model(model_file)

st.title("✊ ✋ ✌️ Rock • Paper • Scissors")
st.caption("Fastai model inference • Upload images and get predictions")

# ---------- Single Image Page ----------
if page == "Single Image":
    file = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","webp"])
    if file is not None:
        img = Image.open(file)
        st.image(img, caption="Uploaded image", use_column_width=True)
        with st.spinner("Predicting..."):
            label, conf, probs = predict_image(learn, img)
        st.subheader(f"Prediction: **{label}**")
        st.write(f"Confidence: **{conf*100:.2f}%**")

        if classes is not None:
            prob_table = pd.DataFrame({
                "class": classes,
                "probability": [float(p) for p in probs],
                "percent": [round(float(p)*100, 2) for p in probs],
            }).sort_values("probability", ascending=False)
            st.dataframe(prob_table, use_container_width=True)

# ---------- Batch Page ----------
else:
    files = st.file_uploader("Upload multiple images",
                             type=["jpg","jpeg","png","bmp","webp"],
                             accept_multiple_files=True)
    if files:
        rows = []
        for f in files:
            try:
                img = Image.open(f)
                label, conf, probs = predict_image(learn, img)
                rows.append({
                    "filename": f.name,
                    "prediction": label,
                    "confidence": round(conf*100, 2)
                })
            except Exception as e:
                rows.append({
                    "filename": f.name,
                    "prediction": f"Error: {e}",
                    "confidence": ""
                })
        df = pd.DataFrame(rows)
        st.success(f"Predicted {len(rows)} image(s)")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download results as CSV", csv,
                           "predictions.csv", "text/csv")

st.markdown("---")
st.caption("Built with fastai + Streamlit. Drop your `rps.pkl` next to `app.py`.")
