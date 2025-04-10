# streamlit_app/app.py

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import sys
import pathlib

# Add scripts folder to import path
sys.path.append(str(pathlib.Path(__file__).parent.parent / "scripts"))

from graphology_features import extract_graphology_features

# Trait labels
traits = ["Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness"]

# CNN model architecture
class PersonalityCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

# Load the trained CNN model
@st.cache_resource
def load_model():
    model = PersonalityCNN()
    model_path = pathlib.Path(__file__).parent.parent / "models" / "personality_cnn.pth"
    model.load_state_dict(torch.load(str(model_path), map_location=torch.device("cpu")))
    model.eval()
    return model

# Predict personality using CNN
def predict_cnn(image, model):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        predicted = np.argmax(probs)
    return traits[predicted], probs

# Generate reasoning from graphology features
def interpret_graphology(features):
    reasoning = []
    if features["Letter Size"] == "Small":
        reasoning.append("Small letters → Focused → Conscientiousness ↑")
    elif features["Letter Size"] == "Large":
        reasoning.append("Large letters → Expressive → Extraversion ↑")

    if features["Letter Slant"] == "Right":
        reasoning.append("Right slant → Sociable → Extraversion ↑")
    elif features["Letter Slant"] == "Left":
        reasoning.append("Left slant → Reserved → Introversion ↑")

    if features["Baseline"] == "Rising":
        reasoning.append("Rising baseline → Optimism → Openness ↑")
    elif features["Baseline"] == "Falling":
        reasoning.append("Falling baseline → Fatigue → Neuroticism ↑")

    if features["Pen Pressure"] == "Heavy":
        reasoning.append("Heavy pressure → Determined → Conscientiousness ↑")
    elif features["Pen Pressure"] == "Light":
        reasoning.append("Light pressure → Sensitive → Agreeableness ↑")

    if features["Word Spacing"] == "Wide":
        reasoning.append("Wide spacing → Independent → Openness ↑")
    elif features["Word Spacing"] == "Narrow":
        reasoning.append("Narrow spacing → Close relationships → Agreeableness ↑")

    return reasoning

# Streamlit UI
st.set_page_config(page_title="🧠 Hybrid Personality Predictor", layout="centered")
st.title("🧠 Personality Predictor from Handwriting (Hybrid)")
st.write("Upload a handwriting image to analyze personality using both AI and Graphology.")

uploaded_file = st.file_uploader("📄 Upload Handwriting Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="🖼️ Uploaded Sample", use_column_width=True)

    if st.button("🔍 Analyze Personality"):
        with st.spinner("Running hybrid analysis..."):

            # Load model and run prediction
            model = load_model()
            pred_trait, scores = predict_cnn(image, model)

            # Extract graphology features and reasoning
            graph_features = extract_graphology_features(uploaded_file)
            reasoning = interpret_graphology(graph_features)

        # Display results
        st.subheader("🔮 CNN Prediction")
        st.success(f"**Predicted Trait:** {pred_trait}")
        st.write("📊 Confidence Scores:")
        for i, t in enumerate(traits):
            st.write(f"- {t}: {scores[i]*100:.2f}%")

        st.subheader("🧠 Graphology Features")
        for k, v in graph_features.items():
            st.write(f"- {k}: {v}")

        st.subheader("💡 Interpretation from Handwriting")
        if reasoning:
            for r in reasoning:
                st.write(f"✔️ {r}")
        else:
            st.write("⚠️ Not enough features to interpret handwriting.")
