# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:22:03 2026

@author: HP
"""

import cv2
import numpy as np

from pathlib import Path


ROOT = Path(r"D:\Competition_26_Code\Proj_5\BottomV_Model")

IMAGE_ROOT = ROOT / "MobileNetDataset" / "train"
OUTPUT_DIR = ROOT / "MobileNetV3_Results" / "TF_SavedModel"

files = sorted(IMAGE_ROOT.glob("FAULT/*.jpg"))

image_path = str(files[0])
img = cv2.imread(image_path)

if img is None:
    raise ValueError(f"Cannot read image: {image_path}")

# Convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize to model input dimensions: Width=480, Height=384
img = cv2.resize(
    img,
    (480, 384),
    interpolation=cv2.INTER_LINEAR
)

# Convert to float32 [0, 1]
img = img.astype(np.float32) / 255.0