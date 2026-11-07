# Python – Computer Vision and Machine Learning Pipeline

## Overview

The Python module contains the computer vision, image processing, geometrical inspection, and machine learning development pipeline for MotorVision-EdgeAI.

It is used for YOLO model training and inference, motor feature extraction, geometrical measurements, motor surface analysis, and MobileNetV3 model development.

## Functional Modules

### 1. YOLO Object Detection and localization 
- Bottom View object detection
- Motor component localization
- Detection-based image validation and cropping

### 2. Image Processing
- Motor image cropping and region extraction
- OpenCV-based image processing
- Image Pre-processing
- Geometrical transformations

### 3. Geometrical Inspection

**Bottom View**
- Outer Diameter (OD) measurement
- Concentricity measurement
- Reference-based pixel-to-mm calibration

### 4. Surface Inspection
- Blow-hole detection
- Image feature extraction
- Surface analysis

### 5. MobileNetV3
- Dataset preparation
- Model training and validation
- Model evaluation
- Motor health-status classification
- TensorFlow Lite model conversion
- INT8 quantization for embedded deployment

## Development Environment

- Python
- OpenCV
- Ultralytics YOLO
- PyTorch
- TensorFlow / TensorFlow Lite
- NumPy
- Pandas

## Role in the Overall System

The Python pipeline is used for model development, training, testing, image processing, and geometrical inspection.

The trained and optimized AI models are subsequently deployed on the Renesas RA8P1 Titan Mini for Edge AI inference and motor health-status classification.
