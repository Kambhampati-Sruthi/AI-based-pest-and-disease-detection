import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

st.set_page_config(page_title="PlantVillage MobileNetV2", layout="centered")

st.title("PlantVillage Disease Classifier (MobileNetV2)")

@st.cache_resource
def load_model(model_dir):
    model = tf.keras.models.load_model("best.keras")
    labels = json.load(open("labels.json"))
    class_names = labels["class_names"]
    return model, class_names

model_dir = st.text_input("Path to SavedModel directory", value="best.keras")

if st.button("Load Model"):
    try:
        model, class_names = load_model(model_dir)
        st.success(f"Loaded model with {len(class_names)} classes.")
        st.session_state["model_loaded"] = True
        st.session_state["class_names"] = class_names
        st.session_state["model"] = model
        
    except Exception as e:
        st.error(f"Failed to load: {e}")

if st.session_state.get("model_loaded", False):
    uploaded = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Input", use_container_width=True)
        model = st.session_state['model']
        img_h = model.input_shape[1]
        img_w = model.input_shape[2]
        img = image.resize((img_w, img_h))
        x = np.asarray(img, dtype=np.float32)  # raw [0..255] RGB
        x = np.expand_dims(x, 0)

        with st.spinner("Predicting..."):
            preds = model.predict(x)[0]
            idx = int(np.argmax(preds))
            prob = float(preds[idx])
            st.subheader(f"Prediction: **{st.session_state['class_names'][idx]}**")
            st.write(f"Confidence: {prob:.2%}")
            st.bar_chart(preds)

