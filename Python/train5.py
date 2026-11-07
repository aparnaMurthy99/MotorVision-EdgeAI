# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 18:50:37 2026

@author: HP
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# Load pretrained YOLOv8 Nano
model = YOLO("yolov8n.pt")

# Fine-tune BottomView model
results = model.train(
    data="Proj_5.yaml",
    epochs=100,
    imgsz=[384, 480],
    multi_scale=True,
    batch=4,
    device="cpu",
    workers=0,
    name="BottomView"
)

print("BottomView training completed!")