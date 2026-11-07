# RA8P1 Titan Mini – Embedded Edge AI 
## Overview

The Renesas RA8P1 Titan Mini acts as the main Edge AI processing unit in the MotorVision-EdgeAI system.

It receives image data from the ESP32-S3 through SPI, performs on-device AI inference, and generates the motor health status.

## Hardware and Software

- Renesas RA8P1 Titan Mini
- Renesas Flexible Software Package (FSP)
- Hardware Abstraction Layer (HAL)
- RT-Thread delay and console logging APIs
- SPI communication interface
- TensorFlow Lite Micro
- INT8 quantized neural network model
- Vela compiler for Ethos-U55 model compilation
- J-Link for debugging and programming

## Functional Modules

### 1. SPI Communication
- SPI Master implementation
- Communication with ESP32-S3 acting as SPI Slave
- Chunk-based RGB888 image reception
- Frame transfer and data validation

### 2. Image Processing
- Received image buffer handling
- Input image preparation for AI inference
- Model input tensor preparation

### 3. Edge AI Inference
- TensorFlow Lite Micro model integration
- INT8 model quantization
- Ethos-U55 model compilation using Vela
- On-device model inference
- Inference output processing

### 4. Status Indication
- LED-based system status indication

## Role in the Overall System

The RA8P1 Titan Mini performs the embedded inference stage of MotorVision-EdgeAI. It receives the camera image from the ESP32-S3, executes the deployed AI model locally, processes the inference output, and provides system status indication.
