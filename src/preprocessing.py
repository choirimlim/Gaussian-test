import cv2
import numpy as np
from pathlib import Path

def remove_shadows(image):
    """그림자 제거."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.merge((l, a, b))

def correct_colors(image):
    """색상 보정."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    return cv2.merge((l, a, b))

def preprocess_images(input_dir, output_dir):
    """이미지 전처리 수행."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.glob("*.jpg"):
        image = cv2.imread(str(img_path))
        no_shadow = remove_shadows(image)
        corrected = correct_colors(no_shadow)
        cv2.imwrite(str(output_dir / img_path.name), corrected)

    print(f"Preprocessed images saved to {output_dir}")
