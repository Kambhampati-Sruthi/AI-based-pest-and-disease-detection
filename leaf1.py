import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from gtts import gTTS
import base64
from datetime import datetime

st.set_page_config(page_title="PlantVillage Disease Classifier", layout="centered")
st.title("🌿 PlantVillage Disease Classifier (MobileNetV2)")

# 🌐 Language selection
language = st.selectbox("Choose language for precautions", ["English", "Telugu", "Hindi"])

# 🧠 Load model and labels
@st.cache_resource
def load_model(model_dir):
    model = tf.keras.models.load_model(model_dir)
    labels = json.load(open("labels.json"))
    class_names = labels["class_names"]
    return model, class_names

# 🔈 Voice playback
def speak_precaution(text, lang_code):
    lang_map = {"English": "en", "Telugu": "te", "Hindi": "hi"}
    tts = gTTS(text, lang=lang_map.get(lang_code, "en"))
    tts.save("precaution.mp3")
    st.audio("precaution.mp3", format="audio/mp3")

# 📥 Report download
def generate_report(label, confidence, precaution):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""🌿 Plant Disease Report 🌿

Prediction: {label}
Confidence: {confidence:.2%}
Time: {timestamp}

Precaution Advice:
{precaution}
"""
    b64 = base64.b64encode(report.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{label}_report.txt">📥 Download Report</a>'
    st.markdown(href, unsafe_allow_html=True)

# 🕘 Prediction history
if "history" not in st.session_state:
    st.session_state["history"] = []

# 📋 Precaution messages for all classes
class_names = [
    "Apple_Black Rot", "Apple_Cedar Rust", "Apple_Healthy", "Apple_Scab",
    "Bell Pepper_Healthy", "Bell Pepper_Healthy _Bacterial Spot",
    "Cherry_Powdert Mildew", "Cherry_Powdert Mildew_Healthy",
    "Corn Maize_Cercospora Leaf Spot", "Corn Maize_Healthy_Common Rust",
    "Corn Maize_Healthy_Northern Blight", "Grape_Black Rot", "Grape_Healthy",
    "Grape_Leaf Blight", "Grape_esca_Black Mealel", "Peach_Bacterial Spot",
    "Peach_Healthy", "Potato_Early Blight", "Potato_Healthy", "Potato_Late Blight",
    "Strawberry", "Strawberry_Healthy", "Strawberry_Leaf Scorch",
    "Tomato_Bacterial Spot", "Tomato_Early Blight", "Tomato_Healthy",
    "Tomato_Late Blight", "Tomato_Septorial Leaf Spot", "Tomato_Yellow Leaf Curl Virus"
]

precautions = {
    cls: {
        "English": f"No specific advice for {cls.replace('_', ' ')}. Please consult an agricultural expert.",
        "Telugu": f"{cls.replace('_', ' ')} కోసం ప్రత్యేక సలహా లేదు. దయచేసి వ్యవసాయ నిపుణుడిని సంప్రదించండి.",
        "Hindi": f"{cls.replace('_', ' ')} के लिए कोई विशिष्ट सलाह नहीं है। कृपया कृषि विशेषज्ञ से संपर्क करें।"
    }
    for cls in class_names
}

# 📂 Load model
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

# 📸 Prediction
if st.session_state.get("model_loaded", False):
    uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Input", use_container_width=True)
        model = st.session_state['model']
        img_h = model.input_shape[1]
        img_w = model.input_shape[2]
        img = image.resize((img_w, img_h))
        x = np.asarray(img, dtype=np.float32)
        x = np.expand_dims(x, 0)

        with st.spinner("Predicting..."):
            preds = model.predict(x)[0]
            idx = int(np.argmax(preds))
            prob = float(preds[idx])
            label = st.session_state['class_names'][idx]

            st.subheader(f"Prediction: **{label}**")
            st.write(f"Confidence: {prob:.2%}")
            st.bar_chart(preds)

            precaution = precautions.get(label, {}).get(language)
            st.success(f"Precaution: {precaution}")
            speak_precaution(precaution, language)
            generate_report(label, prob, precaution)

            st.session_state["history"].append((label, prob, precaution))

    with st.expander("📜 Prediction History"):
        for i, (lbl, conf, prec) in enumerate(st.session_state["history"]):
            st.write(f"{i+1}. **{lbl}** ({conf:.2%}) → {prec}")
