# Rock • Paper • Scissors — Streamlit (fastai) on Hugging Face Spaces

A tiny Streamlit app that loads a **fastai** exported model (`rps.pkl`) to classify images as
**rock**, **paper**, or **scissors**.

## How to use

1. Create a new Space on Hugging Face:
   - **Create -> New Space**
   - SDK: **Streamlit**
   - Hardware: **CPU Basic** is sufficient

2. Add these files to your Space:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `rps.pkl` (your trained fastai model)

3. The app exposes two pages via the sidebar:
   - **Single Image**: upload one image and get prediction + probabilities
   - **Batch (multiple images)**: upload several images at once and download a CSV of results

### Notes
- The app expects `rps.pkl` in the repo root. You can change the path in the sidebar if needed.
- `load_learner` works with models exported by `learn.export()`.
- The app runs entirely on CPU.
