import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
import os
from io import BytesIO

# -------------------- CONFIG --------------------
MODEL_URL = "https://huggingface.co/username/resnet18-ai-detector.pth"
MODEL_FILENAME = "resnet18-ai-detector.pth"
HF_TOKEN = "ababaababaababb"  # 👈 Replace with your token
LABELS = ["AI-Generated", "Real Image"]

st.set_page_config(page_title="AI vs Real Image Detector", layout="centered")
st.markdown("""
    <style>
    body {
        background-color: #fef6f9;
    }
    .stButton button {
        background-color: #dab6fc;
        color: black;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- DOWNLOAD MODEL --------------------
def download_model():
    if os.path.exists(MODEL_FILENAME):
        return
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.get(MODEL_URL, headers=headers)
    if response.status_code == 200:
        with open(MODEL_FILENAME, "wb") as f:
            f.write(response.content)
    else:
        st.error(f"❌ Failed to download model. Status Code: {response.status_code}")
        st.stop()

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    download_model()
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_FILENAME, map_location="cpu"))
    model.eval()
    return model

# -------------------- PREDICT FUNCTION --------------------
def predict_image(image, model):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
        return LABELS[predicted], confidence

# -------------------- UI --------------------
st.title("🧠 AI vs Real Image Detector")
st.markdown("Upload an image and the system will detect if it's AI-generated or a real photograph.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        model = load_model()
        label, confidence = predict_image(image, model)

        st.success(f"✅ Prediction: **{label}** with **{confidence*100:.2f}%** confidence")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
else:
    st.warning("Please upload an image.")

