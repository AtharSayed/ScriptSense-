# scripts/predictor.py

import sys
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# Import graphology feature extractor
from graphology_features import extract_graphology_features

# Trait labels (order matches training)
traits = ["Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness"]

# CNN Model (same as used in training)
class PersonalityCNN(nn.Module):
    def __init__(self):
        super(PersonalityCNN, self).__init__()
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

# CNN prediction + scores
def cnn_predict(image_path, model_path='models/personality_cnn.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PersonalityCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()

    return traits[pred_idx], probs

# Rule-based graphology interpretation
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
        reasoning.append("Falling baseline → Fatigue/Low mood → Neuroticism ↑")

    if features["Pen Pressure"] == "Heavy":
        reasoning.append("Heavy pressure → Determined → Conscientiousness ↑")
    elif features["Pen Pressure"] == "Light":
        reasoning.append("Light pressure → Sensitive → Agreeableness ↑")

    if features["Word Spacing"] == "Wide":
        reasoning.append("Wide spacing → Independent → Openness ↑")
    elif features["Word Spacing"] == "Narrow":
        reasoning.append("Narrow spacing → Need for closeness → Agreeableness ↑")

    return reasoning

# 🔁 Main
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predictor.py path_to_image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        sys.exit(1)

    print(f"\n🖼️ Image: {os.path.basename(image_path)}")

    # 1. CNN Prediction
    trait, scores = cnn_predict(image_path)
    print(f"\n🔮 CNN Predicted Trait: {trait}")
    print("📊 Confidence Scores:")
    for i, t in enumerate(traits):
        print(f"  {t}: {scores[i]*100:.2f}%")

    # 2. Graphology Analysis
    print("\n📎 Graphology-Based Features:")
    features = extract_graphology_features(image_path)
    for k, v in features.items():
        print(f"  {k}: {v}")

    # 3. Reasoning
    print("\n💡 Interpretation Based on Graphology:")
    reasoning = interpret_graphology(features)
    if reasoning:
        for r in reasoning:
            print(f"  - {r}")
    else:
        print("  ⚠️ Not enough visual cues to interpret.")

    print("\n✅ Hybrid analysis complete.\n")
