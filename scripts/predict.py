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
def interpret_graphology(feature_dicts):
    reasoning = {}

    # Convert list of dicts to plain dict for easier access
    features = {f["Attribute"]: f["Writing Category"] for f in feature_dicts}

    if features.get("Letter Size") == "Small":
        reasoning["Conscientiousness"] = "Small letters → Focused → Conscientiousness ↑"
    elif features.get("Letter Size") == "Large":
        reasoning["Extraversion"] = "Large letters → Expressive → Extraversion ↑"

    if features.get("Letter Slant") == "Right":
        reasoning["Extraversion_2"] = "Right slant → Sociable → Extraversion ↑"
    elif features.get("Letter Slant") == "Left":
        reasoning["Introversion"] = "Left slant → Reserved → Introversion ↑"

    if features.get("Baseline") == "Rising":
        reasoning["Openness"] = "Rising baseline → Optimism → Openness ↑"
    elif features.get("Baseline") == "Falling":
        reasoning["Neuroticism"] = "Falling baseline → Fatigue/Low mood → Neuroticism ↑"

    if features.get("Pen Pressure") == "Heavy":
        reasoning["Conscientiousness_2"] = "Heavy pressure → Determined → Conscientiousness ↑"
    elif features.get("Pen Pressure") == "Light":
        reasoning["Agreeableness"] = "Light pressure → Sensitive → Agreeableness ↑"

    if features.get("Word Spacing") == "Wide":
        reasoning["Openness_2"] = "Wide spacing → Independent → Openness ↑"
    elif features.get("Word Spacing") == "Narrow":
        reasoning["Agreeableness_2"] = "Narrow spacing → Need for closeness → Agreeableness ↑"

    return list(reasoning.values())

# 🔁 Main
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path_to_image.jpg")
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
    for feat in features:
        print(f"  {feat['Attribute']}: {feat['Writing Category']} → {feat['Psychological Personality Behavior']}")

    # 3. Reasoning
    print("\n💡 Interpretation Based on Graphology:")
    reasoning = interpret_graphology(features)
    if reasoning:
        for r in reasoning:
            print(f"  - {r}")
    else:
        print("  ⚠️ Not enough visual cues to interpret.")

    print("\n✅ Hybrid analysis complete.\n")
