import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from gtts import gTTS
import matplotlib.pyplot as plt
import base64
from datetime import datetime
import pandas as pd

# 🌿 Page setup
st.set_page_config(page_title="PlantVillage Disease Classifier", layout="centered")
st.title("🌿 PlantVillage Disease Classifier (MobileNetV2)")

# 🌐 Language selection
language = st.selectbox("Choose language for precautions", ["English", "Telugu", "Hindi"])

# 🧠 Load model and labels
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best.keras")
    labels = json.load(open("labels.json"))
    class_names = labels["class_names"]
    return model, class_names

model, class_names = load_model()

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

# 📋 Precautions dictionary (add all 29 classes here)

precautions = {
    "Apple_Black Rot": {
        "English": "Prune infected branches and apply fungicide.",
        "Telugu": "సంఖ్యలో ఉన్న కొమ్మలను తొలగించి ఫంగిసైడ్‌ను ఉపయోగించండి.",
        "Hindi": "संक्रमित शाखाओं को काटें और फफूंदनाशक लगाएं।"
    },
    "Apple_Cedar Rust": {
        "English": "Remove nearby juniper trees and spray sulfur-based fungicide.",
        "Telugu": "జునిపర్ చెట్లను తొలగించి సల్ఫర్ ఆధారిత ఫంగిసైడ్‌ను స్ప్రే చేయండి.",
        "Hindi": "पास के जुनिपर पेड़ों को हटाएं और सल्फर आधारित फफूंदनाशक छिड़कें।"
    },
    "Apple_Healthy": {
        "English": "Maintain regular pruning and balanced fertilization.",
        "Telugu": "క్రమం తప్పకుండా కత్తిరించడం మరియు సమతుల్య ఎరువులు ఇవ్వడం కొనసాగించండి.",
        "Hindi": "नियमित छंटाई और संतुलित उर्वरक देना जारी रखें।"
    },
    "Apple_Scab": {
        "English": "Apply fungicide during early spring and remove fallen leaves.",
        "Telugu": "వసంత కాలంలో ఫంగిసైడ్‌ను ఉపయోగించి, పడ్డ ఆకులను తొలగించండి.",
        "Hindi": "वसंत ऋतु में फफूंदनाशक लगाएं और गिरे हुए पत्तों को हटाएं।"
    },
    "Bell Pepper_Healthy": {
        "English": "Monitor for pests and maintain proper irrigation.",
        "Telugu": "కీటకాలను గమనించి సరైన నీటిపారుదల కొనసాగించండి.",
        "Hindi": "कीटों की निगरानी करें और उचित सिंचाई बनाए रखें।"
    },
    "Bell Pepper_Healthy _Bacterial Spot": {
        "English": "Use copper-based sprays and avoid overhead watering.",
        "Telugu": "కాపర్ ఆధారిత స్ప్రేలు ఉపయోగించి, పై నుండి నీరు పోయడం నివారించండి.",
        "Hindi": "तांबा आधारित स्प्रे का उपयोग करें और ऊपर से पानी देना बंद करें।"
    },
    "Cherry_Powdert Mildew": {
        "English": "Apply sulfur fungicide and improve air circulation.",
        "Telugu": "సల్ఫర్ ఫంగిసైడ్‌ను ఉపయోగించి గాలి ప్రసరణను మెరుగుపరచండి.",
        "Hindi": "सल्फर फफूंदनाशक लगाएं और वायु संचार सुधारें।"
    },
    "Cherry_Powdert Mildew_Healthy": {
        "English": "Continue monitoring and maintain spacing between trees.",
        "Telugu": "పరిశీలన కొనసాగించి చెట్ల మధ్య అంతరాన్ని ఉంచండి.",
        "Hindi": "निरंतर निगरानी रखें और पेड़ों के बीच उचित दूरी बनाए रखें।"
    },
    "Corn Maize_Cercospora Leaf Spot": {
        "English": "Rotate crops and use resistant hybrids.",
        "Telugu": "పంటలను మారుస్తూ సాగు చేసి ప్రతిఘటించే హైబ్రిడ్స్‌ను ఉపయోగించండి.",
        "Hindi": "फसल चक्र अपनाएं और प्रतिरोधी किस्मों का उपयोग करें।"
    },
    "Corn Maize_Healthy_Common Rust": {
        "English": "Monitor for rust and apply fungicide if needed.",
        "Telugu": "రస్ట్ కోసం పరిశీలించి అవసరమైతే ఫంగిసైడ్‌ను ఉపయోగించండి.",
        "Hindi": "रस्ट की निगरानी करें और आवश्यकता होने पर फफूंदनाशक लगाएं।"
    },
    "Corn Maize_Healthy_Northern Blight": {
        "English": "Use disease-free seeds and maintain field hygiene.",
        "Telugu": "రోగం లేని విత్తనాలను ఉపయోగించి పొల శుభ్రతను కొనసాగించండి.",
        "Hindi": "रोग-मुक्त बीजों का उपयोग करें और खेत की सफाई बनाए रखें।"
    },
    "Grape_Black Rot": {
        "English": "Remove infected grapes and apply fungicide.",
        "Telugu": "సంఖ్యలో ఉన్న ద్రాక్షలను తొలగించి ఫంగిసైడ్‌ను ఉపయోగించండి.",
        "Hindi": "संक्रमित अंगूरों को हटाएं और फफूंदनाशक लगाएं।"
    },
    "Grape_Healthy": {
        "English": "Maintain trellis structure and monitor for mildew.",
        "Telugu": "ట్రెలిస్ నిర్మాణాన్ని నిర్వహించి మిల్డ్యూ కోసం పరిశీలించండి.",
        "Hindi": "ट्रेलिस संरचना बनाए रखें और फफूंदी की निगरानी करें।"
    },
    "Grape_Leaf Blight": {
        "English": "Prune affected leaves and apply protective sprays.",
        "Telugu": "బాధిత ఆకులను కత్తిరించి రక్షణ స్ప్రేలు చేయండి.",
        "Hindi": "प्रभावित पत्तों को काटें और सुरक्षात्मक स्प्रे लगाएं।"
    },
    "Grape_esca_Black Mealel": {
        "English": "Remove infected vines and avoid water stress.",
        "Telugu": "సంఖ్యలో ఉన్న ద్రాక్ష తాడులను తొలగించి నీటి ఒత్తిడిని నివారించండి.",
        "Hindi": "संक्रमित बेलों को हटाएं और जल तनाव से बचें।"
    },
    "Peach_Bacterial Spot": {
        "English": "Use copper sprays and avoid wet foliage.",
        "Telugu": "కాపర్ స్ప్రేలు ఉపయోగించి తడి ఆకులను నివారించండి.",
        "Hindi": "तांबा स्प्रे का उपयोग करें और गीली पत्तियों से बचें।"
    },
    "Peach_Healthy": {
        "English": "Continue regular pruning and pest monitoring.",
        "Telugu": "క్రమం తప్పకుండా కత్తిరించడం మరియు కీటక పరిశీలన కొనసాగించండి.",
        "Hindi": "नियमित छंटाई और कीट निगरानी जारी रखें।"
    },
    "Potato_Early Blight": {
        "English": "Apply fungicide and remove infected leaves.",
        "Telugu": "ఫంగిసైడ్‌ను ఉపయోగించి బాధిత ఆకులను తొలగించండి.",
        "Hindi": "फफूंदनाशक लगाएं और संक्रमित पत्तों को हटाएं।"
    },
    "Potato_Healthy": {
        "English": "Maintain soil health and monitor for blight.",
        "Telugu": "మట్టి ఆరోగ్యాన్ని నిర్వహించి బ్లైట్ కోసం పరిశీలించండి.",
        "Hindi": "मिट्टी की सेहत बनाए रखें और ब्लाइट की निगरानी करें।"
    },
    "Potato_Late Blight": {
        "English": "Apply fungicide promptly and avoid overhead irrigation.",
        "Telugu": "ఫంగిసైడ్‌ను వెంటనే ఉపయోగించి పై నుండి నీటిపారుదల నివారించండి.",
        "Hindi": "फफूंदनाशक तुरंत लगाएं और ऊपर से सिंचाई से बचें।"
    },
    "Strawberry": {
        "English": "Use mulch and avoid water splash on leaves.",
        "Telugu": "మల్చ్‌ను ఉపయోగించి ఆకులపై నీటి చిమ్ముటను నివారించండి.",
        "Hindi": "मल्च का उपयोग करें और पत्तों पर पानी के छींटे से बचें।"
    },
    "Strawberry_Healthy": {
        "English": "Maintain spacing and monitor for leaf scorch.",
        "Telugu": "అంతరాన్ని ఉంచి ఆకుల కాలిన లక్షణాల కోసం పరిశీలించండి.",
        "Hindi": "दूरी बनाए रखें और पत्तों के झुलसने की निगरानी करें।"
    },
    "Strawberry_Leaf Scorch": {
        "English": "Remove scorched leaves and improve irrigation.",
        "Telugu": "కాలిన ఆకులను తొలగించి నీటిపారుదల మెరుగుపరచండి.",
        "Hindi": "झुलसे हुए पत्तों को हटाएं और सिंचाई सुधारें।"
    },
    "Tomato_Bacterial Spot": {
    "English": "Use copper-based sprays and avoid leaf wetness.",
    "Telugu": "కాపర్ ఆధారిత స్ప్రేలు ఉపయోగించి ఆకుల తడిని నివారించండి.",
    "Hindi": "तांबा आधारित स्प्रे का उपयोग करें और पत्तों को गीला होने से बचाएं।"
},
"Tomato_Early Blight": {
    "English": "Apply fungicide and remove infected foliage.",
    "Telugu": "ఫంగిసైడ్‌ను ఉపయోగించి బాధిత ఆకులను తొలగించండి.",
    "Hindi": "फफूंदनाशक लगाएं और संक्रमित पत्तों को हटाएं।"
},
"Tomato_Healthy": {
    "English": "Maintain crop rotation and monitor for blight.",
    "Telugu": "పంటల మార్పిడి కొనసాగించి బ్లైట్ కోసం పరిశీలించండి.",
    "Hindi": "फसल चक्र बनाए रखें और ब्लाइट की निगरानी करें।"
},
"Tomato_Late Blight": {
    "English": "Use resistant varieties and apply fungicide during cool, wet weather.",
    "Telugu": "ప్రతిఘటించే రకాలను ఉపయోగించి చల్లని, తడి వాతావరణంలో ఫంగిసైడ్‌ను స్ప్రే చేయండి.",
    "Hindi": "प्रतिरोधी किस्मों का उपयोग करें और ठंडे, नम मौसम में फफूंदनाशक लगाएं।"
},
"Tomato_Septorial Leaf Spot": {
    "English": "Remove infected leaves and avoid overhead watering.",
    "Telugu": "బాధిత ఆకులను తొలగించి పై నుండి నీరు పోయడం నివారించండి.",
    "Hindi": "संक्रमित पत्तों को हटाएं और ऊपर से पानी देना बंद करें।"
},
"Tomato_Yellow Leaf Curl Virus": {
    "English": "Control whiteflies and use virus-resistant varieties.",
    "Telugu": "తెల్లతెగులు నియంత్రించి వైరస్-ప్రతిఘటించే రకాలను ఉపయోగించండి.",
    "Hindi": "सफेद मक्खियों को नियंत्रित करें और वायरस-प्रतिरोधी किस्मों का उपयोग करें।"
}
}

# 📁 File upload
st.subheader("📁 Upload Leaf Image")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# 🧠 Prediction
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_container_width=True)

    model = st.session_state.get("model", model)
    class_names = st.session_state.get("class_names", class_names)

    img_h, img_w = model.input_shape[1], model.input_shape[2]
    image_resized = image.resize((img_w, img_h))
    x = np.asarray(image_resized, dtype=np.float32) / 255.0
    x = np.expand_dims(x, 0)

    with st.spinner("🔍 Predicting..."):
        preds = model.predict(x)[0]
        idx = int(np.argmax(preds))
        prob = float(preds[idx])
        label = class_names[idx]

        st.subheader(f"🧠 Prediction: **{label}**")
        st.write(f"Confidence: {prob:.2%}")

        # 📊 Confidence bar chart
        st.subheader("📊 Confidence Scores")
        st.bar_chart(preds)

        # 🛡️ Precaution Advice
        precaution = precautions.get(label, {}).get(language, "Precaution not available.")
        st.subheader("🛡️ Precaution Advice")
        st.write(precaution)
        speak_precaution(precaution, language)

        # 📥 Downloadable Report
        generate_report(label, prob, precaution)

        # 🕘 Save to History
        st.session_state["history"].append({
            "label": label,
            "confidence": prob,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

