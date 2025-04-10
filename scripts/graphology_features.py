# src/graphology_features.py

import cv2
import numpy as np
from PIL import Image

def detect_letter_size(gray_img):
    contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(cnt)[3] for cnt in contours if cv2.contourArea(cnt) > 10]
    if not heights:
        return "Unknown"
    avg_height = np.mean(heights)
    if avg_height < 15:
        return "Small"
    elif avg_height < 30:
        return "Medium"
    else:
        return "Large"

def detect_letter_slant(gray_img):
    edges = cv2.Canny(gray_img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=30, maxLineGap=10)
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            if -90 < angle < 90:  # valid slant
                angles.append(angle)
    if not angles:
        return "Vertical"
    avg_angle = np.mean(angles)
    if avg_angle < -5:
        return "Left"
    elif avg_angle > 5:
        return "Right"
    else:
        return "Vertical"

def detect_pen_pressure(gray_img):
    mean_intensity = np.mean(gray_img)
    if mean_intensity < 80:
        return "Heavy"
    elif mean_intensity < 160:
        return "Medium"
    else:
        return "Light"

def detect_baseline(gray_img):
    projection = np.sum(255 - gray_img, axis=1)  # sum of ink on each row
    indices = np.nonzero(projection > np.max(projection) * 0.3)[0]
    if len(indices) < 2:
        return "Unknown"
    poly = np.polyfit(indices, range(len(indices)), 1)
    slope = poly[0]
    if slope > 0.2:
        return "Rising"
    elif slope < -0.2:
        return "Falling"
    else:
        return "Straight"

def detect_word_spacing(gray_img):
    horizontal_proj = np.sum(255 - gray_img, axis=0)
    gaps = []
    gap_len = 0
    for val in horizontal_proj:
        if val < 10:
            gap_len += 1
        elif gap_len > 0:
            gaps.append(gap_len)
            gap_len = 0
    if not gaps:
        return "Unknown"
    avg_gap = np.mean(gaps)
    if avg_gap < 5:
        return "Narrow"
    elif avg_gap < 15:
        return "Normal"
    else:
        return "Wide"

def extract_graphology_features(image_path):
    img = Image.open(image_path).convert("L")
    gray = np.array(img)

    return {
        "Letter Size": detect_letter_size(gray),
        "Letter Slant": detect_letter_slant(gray),
        "Pen Pressure": detect_pen_pressure(gray),
        "Baseline": detect_baseline(gray),
        "Word Spacing": detect_word_spacing(gray),
    }
