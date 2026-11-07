# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:04:21 2026

@author: HP
"""


import os
import torch
import torch.nn as nn
import onnx

from pathlib import Path
from torchvision.models import mobilenet_v3_small


# ==================================================
# 1. PATHS
# ==================================================

PROJECT_ROOT = Path(
    r"D:\Competition_26_Code\Proj_5\BottomV_Model"
)

BEST_MODEL_PATH = (
    PROJECT_ROOT
    / "MobileNetV3_Results"
    / "best_motor_fusion.pth"
)

ONNX_PATH = (
    PROJECT_ROOT
    / "MobileNetV3_Results"
    / "motor_fusion.onnx"
)

os.makedirs(ONNX_PATH.parent, exist_ok=True)


# ==================================================
# 2. INPUT DIMENSIONS
# ==================================================

IMAGE_H = 384
IMAGE_W = 480

NUM_GEOMETRY_FEATURES = 3


# ==================================================
# 3. GEOMETRY NORMALIZATION
#    Values calculated from training CSV
# ==================================================

GEO_MEAN = torch.tensor(
    [99.44135, 22.018229, 15.0],
    dtype=torch.float32
)

GEO_STD = torch.tensor(
    [0.6183304, 0.30580834, 1.0],
    dtype=torch.float32
)


# ==================================================
# 4. IMAGE NORMALIZATION
# ==================================================

IMAGE_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32
)

IMAGE_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32
)


# ==================================================
# 5. ORIGINAL MODEL ARCHITECTURE
# ==================================================

class MotorFusionNet(nn.Module):

    def __init__(self):

        super().__init__()

        # IMAGE BRANCH
        self.backbone = mobilenet_v3_small(
            weights=None
        )

        self.backbone.classifier = nn.Identity()

        # GEOMETRY BRANCH
        self.geometry_branch = nn.Sequential(

            nn.Linear(3, 32),
            nn.ReLU(),

            nn.Dropout(0.15),

            nn.Linear(32, 16),
            nn.ReLU()
        )

        # FEATURE FUSION
        self.fusion = nn.Sequential(

            nn.Linear(576 + 16, 128),
            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(128, 2)
        )

    def forward(self, image, geometry):

        image_features = self.backbone(image)

        geometry_features = self.geometry_branch(
            geometry
        )

        combined = torch.cat(
            [image_features, geometry_features],
            dim=1
        )

        output = self.fusion(combined)

        return output


# ==================================================
# 6. EXPORT WRAPPER
#    Embeds the original preprocessing
# ==================================================

class ExportModel(nn.Module):

    def __init__(self, trained_model):

        super().__init__()

        self.model = trained_model

        self.register_buffer(
            "image_mean",
            IMAGE_MEAN.view(1, 3, 1, 1)
        )

        self.register_buffer(
            "image_std",
            IMAGE_STD.view(1, 3, 1, 1)
        )

        self.register_buffer(
            "geo_mean",
            GEO_MEAN.view(1, 3)
        )

        self.register_buffer(
            "geo_std",
            GEO_STD.view(1, 3)
        )

    def forward(self, image, geometry):

        image = (
            image - self.image_mean
        ) / self.image_std

        geometry = (
            geometry - self.geo_mean
        ) / self.geo_std

        return self.model(image, geometry)


# ==================================================
# 7. LOAD EXISTING TRAINED CHECKPOINT
# ==================================================

print("\nLoading trained checkpoint...")

if not BEST_MODEL_PATH.exists():
    raise FileNotFoundError(BEST_MODEL_PATH)

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

# Support either a direct state_dict or a
# checkpoint dictionary containing a state_dict.

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif "model" in checkpoint and isinstance(
        checkpoint["model"], dict
    ):
        state_dict = checkpoint["model"]

    else:
        state_dict = checkpoint

else:
    raise TypeError(
        "Unexpected checkpoint format."
    )

# Handle DataParallel checkpoints if necessary.

if all(key.startswith("module.") for key in state_dict):
    state_dict = {
        key.replace("module.", "", 1): value
        for key, value in state_dict.items()
    }

model = MotorFusionNet()

model.load_state_dict(
    state_dict,
    strict=True
)

model.eval()

print("Checkpoint loaded successfully.")


# ==================================================
# 8. CREATE EXPORT MODEL
# ==================================================

export_model = ExportModel(model)
export_model.eval()


# ==================================================
# 9. CREATE DUMMY INPUTS
#    Image is RGB float in range 0 to 1.
#    Geometry is in original units.
# ==================================================

dummy_image = torch.rand(
    1, 3, IMAGE_H, IMAGE_W,
    dtype=torch.float32
)

dummy_geometry = torch.tensor(
    [[99.5, 22.0, 15.0]],
    dtype=torch.float32
)


# ==================================================
# 10. VERIFY PREPROCESSING WRAPPER
# ==================================================

print("\nChecking wrapper against original model...")

with torch.no_grad():

    original_image = (
        dummy_image - IMAGE_MEAN.view(1, 3, 1, 1)
    ) / IMAGE_STD.view(1, 3, 1, 1)

    original_geometry = (
        dummy_geometry - GEO_MEAN
    ) / GEO_STD

    original_output = model(
        original_image,
        original_geometry
    )

    wrapper_output = export_model(
        dummy_image,
        dummy_geometry
    )

    difference = torch.max(
        torch.abs(original_output - wrapper_output)
    ).item()

print("Maximum output difference:", difference)

if difference > 1e-5:
    raise RuntimeError(
        "Preprocessing verification failed."
    )

print("Preprocessing verification PASSED.")


# ==================================================
# 11. EXPORT TO ONNX
# ==================================================

print("\nExporting ONNX model...")

torch.onnx.export(

    export_model,

    (dummy_image, dummy_geometry),

    str(ONNX_PATH),

    export_params=True,

    opset_version=17,

    do_constant_folding=True,

    input_names=[
        "image_rgb_0_1",
        "geometry_raw"
    ],

    output_names=[
        "logits"
    ],

    dynamic_axes={

        "image_rgb_0_1": {
            0: "batch_size"
        },

        "geometry_raw": {
            0: "batch_size"
        },

        "logits": {
            0: "batch_size"
        }
    },

    dynamo=False
)


# ==================================================
# 12. VERIFY ONNX FILE
# ==================================================

print("\nChecking ONNX model...")

onnx_model = onnx.load(str(ONNX_PATH))

onnx.checker.check_model(onnx_model)

print("ONNX validation PASSED.")

print("\n===================================")
print("ONNX EXPORT SUCCESSFUL")
print("===================================")

print("Saved at:")
print(ONNX_PATH)

print("\nInput 1: RGB image [N, 3, 384, 480]")
print("Input 2: Raw geometry [N, 3]")
print("Output: 2 logits")

print("\nClass mapping:")
print("0 = NO_FAULT")
print("1 = FAULT")
