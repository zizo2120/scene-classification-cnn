"""Streamlit demo app for the natural-scene image classifier (bonus deliverable).

Run locally with:
    streamlit run app.py

Expects the exported model produced by the notebook (Section "Bonus — Save & Export the
Final Model") to be present as `final_scene_classifier.keras` in this directory. Override
the path with the MODEL_PATH environment variable if it lives elsewhere.
"""
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = os.environ.get("MODEL_PATH", "final_scene_classifier.keras")
IMG_SIZE = (150, 150)
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

st.set_page_config(page_title="Scene Classifier", page_icon="🏔️")


@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def predict(model, image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    array = tf.keras.utils.img_to_array(image)
    batch = np.expand_dims(array, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), dict(zip(CLASS_NAMES, probs.tolist()))


st.title("🏔️ Natural Scene Classifier")
st.write(
    "Upload a photo and the model will classify it as one of: "
    + ", ".join(CLASS_NAMES) + "."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found at `{MODEL_PATH}`. Export it from the notebook first "
        "(see the 'Bonus — Save & Export the Final Model' section), or set the "
        "MODEL_PATH environment variable."
    )
    st.stop()

model = load_model(MODEL_PATH)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Classifying..."):
        predicted_class, confidence, all_probs = predict(model, image)

    st.subheader(f"Prediction: **{predicted_class}** ({confidence:.1%} confidence)")
    st.bar_chart(all_probs)
