# scripts/test_graphology.py

from graphology_features import extract_graphology_features

image_path = "E:/SideProject/project_root/dataset/processed_samples/Openness/IMG_20200215_181852.jpg"
features = extract_graphology_features(image_path)

print("🧠 Extracted Graphology Features:")
for key, value in features.items():
    print(f"  {key}: {value}")
