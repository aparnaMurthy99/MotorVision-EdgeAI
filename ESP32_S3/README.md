# ESP32-S3 – Camera Acquisition and SPI Communication

## Overview

The ESP32-S3 serves as the image acquisition and transmission unit in the MotorVision-EdgeAI system.

It interfaces with the OV3660 camera, captures motor images, performs image preprocessing, and transfers image data to the Renesas RA8P1 Titan Mini for Edge AI processing.

## Hardware

- ESP32-S3 development board (N16R8)
- OV3660 camera module

## Functional Modules

### 1. Camera Acquisition
- Camera initialization and configuration
- Image capture using OV3660
- Frame acquisition and validation

### 2. Image Preprocessing
- RGB888 image conversion
- Image resizing to 480 × 384 pixels (W × H)
- Preparation of raw image data for transmission

### 3. SPI Communication
- SPI Slave implementation
- Communication with RA8P1 Titan Mini acting as SPI Master
- Chunk-based image transmission
- Capture request and trigger handling

## Data Transfer

- Image format: RGB888
- Target resolution: 480 × 384 pixels
- Pixel depth: 24-bit (3 bytes per pixel)
- Frame size: 552,960 bytes
- Communication interface: SPI

## Role in the Overall System

The ESP32-S3 is responsible for camera acquisition and image transfer. AI inference and motor health-status classification are performed on the Renesas RA8P1 Titan Mini.
