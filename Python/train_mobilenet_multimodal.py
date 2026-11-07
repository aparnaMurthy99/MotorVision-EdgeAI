# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:49:44 2026

@author: HP
"""


import os
import glob
import random
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = r"D:\Competition_26_Code\Proj_5\BottomV_Model"

DATASET_ROOT = os.path.join(ROOT, "mobilenetDataset")

CSV_PATHS = {
    "train": os.path.join(ROOT, "GeometricalFeats_train.csv"),
    "val":   os.path.join(ROOT, "GeometricalFeats_val.csv"),
    "test":  os.path.join(ROOT, "GeometricalFeats_test.csv")
}

OUTPUT_DIR = os.path.join(ROOT, "MobileNetV3_Results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_W = 480
IMAGE_H = 384

BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.0001
PATIENCE = 8

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 2. CLASSES AND GEOMETRICAL FEATURES
# ============================================================

CLASS_MAP = {
    "NO_FAULT": 0,
    "FAULT": 1
}

IMAGE_CLASSES = [
    "gearTooth",
    "gearFace",
    "mainFace",
    "flangeFace"
]

# ONLY THREE GEOMETRICAL INPUT FEATURES
GEOMETRY_FEATURES = [
    "OD_Flange",
    "OD_Gear",
    "GearCount"
]

# BlowHoles is deliberately NOT an input.
# Lap_Var is also NOT an input.
# LABEL is the classification target.


# ============================================================
# 3. LOAD AND VALIDATE CSV FILES
# ============================================================

csv_data = {}

for split, csv_path in CSV_PATHS.items():

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    required_columns = [
        "Unit",
        "OD_Flange",
        "OD_Gear",
        "GearCount",
        "BlowHoles",
        "LABEL"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{split} CSV missing columns: {missing_columns}"
        )

    df["Unit"] = (
        df["Unit"].astype(str).str.strip().str.lower()
    )

    df["LABEL"] = pd.to_numeric(
        df["LABEL"], errors="raise"
    ).astype(int)

    if df["Unit"].duplicated().any():
        raise ValueError(
            f"Duplicate Unit entries in {split} CSV"
        )

    if not df["LABEL"].isin([0, 1]).all():
        raise ValueError(
            f"Invalid LABEL values in {split} CSV"
        )

    for col in GEOMETRY_FEATURES:
        df[col] = pd.to_numeric(
            df[col], errors="raise"
        )

    if not np.isfinite(
        df[GEOMETRY_FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"Invalid geometry values in {split} CSV"
        )

    csv_data[split] = df

    print(
        split.upper(),
        "CSV motors:",
        df["Unit"].nunique()
    )


# ============================================================
# 4. CALCULATE GEOMETRY NORMALIZATION
#    TRAIN MOTORS ONLY
# ============================================================

train_csv = csv_data["train"]

GEO_MEAN = train_csv[
    GEOMETRY_FEATURES
].mean().to_numpy(dtype=np.float32)

GEO_STD = train_csv[
    GEOMETRY_FEATURES
].std(ddof=0).to_numpy(dtype=np.float32)


# PRINT NORMALIZATION VALUES
print("GEO_MEAN:", GEO_MEAN)
print("GEO_STD:", GEO_STD)

# Avoid division by zero
GEO_STD[GEO_STD == 0] = 1.0

print("\nTRAIN geometry normalization:")

for name, mean, std in zip(
    GEOMETRY_FEATURES, GEO_MEAN, GEO_STD
):
    print(
        f"{name}: mean={mean:.5f}, std={std:.5f}"
    )


# ============================================================
# 5. CUSTOM PYTORCH DATASET
# ============================================================

class MotorDataset(Dataset):

    def __init__(self, split):

        self.samples = []

        csv_df = csv_data[split]

        label_map = dict(
            zip(csv_df["Unit"], csv_df["LABEL"])
        )

        geometry_map = {
            row["Unit"]: row[GEOMETRY_FEATURES].to_numpy(
                dtype=np.float32
            )
            for _, row in csv_df.iterrows()
        }

        for class_name, folder_label in CLASS_MAP.items():

            folder = os.path.join(
                DATASET_ROOT,
                split,
                class_name
            )

            if not os.path.isdir(folder):
                raise FileNotFoundError(folder)

            image_paths = []

            for ext in [
                "*.jpg", "*.jpeg", "*.png", "*.bmp"
            ]:
                image_paths.extend(
                    glob.glob(os.path.join(folder, ext))
                )

            for image_path in sorted(image_paths):

                filename = os.path.basename(image_path)

                # Example:
                # bv42_bv42_flangeFace.jpg
                unit = filename.split("_")[0].lower()

                if unit not in label_map:
                    raise ValueError(
                        f"{unit} missing from {split} CSV"
                    )

                csv_label = int(label_map[unit])

                # Check CSV label against folder label
                if csv_label != folder_label:
                    raise ValueError(
                        f"Label mismatch: {filename}, "
                        f"folder={folder_label}, "
                        f"CSV={csv_label}"
                    )

                geometry = geometry_map[unit]

                self.samples.append(
                    (
                        image_path,
                        geometry,
                        csv_label,
                        unit
                    )
                )

        # ----------------------------------------
        # VERIFY FOUR CROPS PER MOTOR
        # ----------------------------------------

        motor_counts = {}

        for path, geometry, label, unit in self.samples:
            motor_counts[unit] = (
                motor_counts.get(unit, 0) + 1
            )

        incorrect = {
            unit: count
            for unit, count in motor_counts.items()
            if count != 4
        }

        if incorrect:
            raise ValueError(
                f"{split}: motors without exactly 4 crops: "
                f"{incorrect}"
            )

        print(
            f"\n{split.upper()}: "
            f"{len(motor_counts)} motors, "
            f"{len(self.samples)} images"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, geometry, label, unit = self.samples[index]

        # ----------------------------------------
        # IMAGE INPUT
        # ----------------------------------------

        image = cv2.imread(path)

        if image is None:
            raise ValueError(
                f"Cannot read image: {path}"
            )

        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

        if image.shape != (IMAGE_H, IMAGE_W, 3):
            raise ValueError(
                f"Incorrect image dimensions: "
                f"{path}, {image.shape}"
            )

        # HWC -> CHW
        image = torch.from_numpy(
            image.copy()
        ).permute(2, 0, 1).float() / 255.0

        # ImageNet normalization
        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        image = (image - mean) / std

        # ----------------------------------------
        # GEOMETRY INPUT
        # ----------------------------------------

        geometry = (
            geometry - GEO_MEAN
        ) / GEO_STD

        geometry = torch.tensor(
            geometry,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        #return image, geometry, label
        return image, geometry, label, unit


# ============================================================
# 6. CREATE DATASETS AND LOADERS
# ============================================================

train_dataset = MotorDataset("train")
val_dataset = MotorDataset("val")
test_dataset = MotorDataset("test")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 7. MOBILE NET V3 + GEOMETRY FUSION MODEL
# ============================================================

class MotorFusionNet(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------
        # IMAGE BRANCH
        # ----------------------------------------

        self.backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        )

        # Remove original ImageNet classifier
        self.backbone.classifier = nn.Identity()

        # Output: 576 image features

        # ----------------------------------------
        # GEOMETRY BRANCH
        # ----------------------------------------

        self.geometry_branch = nn.Sequential(

            nn.Linear(3, 32),
            nn.ReLU(),

            nn.Dropout(0.15),

            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Output: 16 geometry features

        # ----------------------------------------
        # FEATURE FUSION
        # ----------------------------------------

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


# ============================================================
# 8. INITIALIZE MODEL AND OPTIMIZER
# ============================================================

model = MotorFusionNet().to(DEVICE)

# Calculate class weights from TRAIN motors
motor_labels = (
    train_csv.drop_duplicates("Unit")["LABEL"].to_numpy()
)

class_counts = np.bincount(
    motor_labels,
    minlength=2
)

if np.any(class_counts == 0):
    raise ValueError(
        "TRAIN dataset must contain both classes."
    )

class_weights = (
    len(motor_labels) / (2.0 * class_counts)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

print("\nTRAIN motor class counts:", class_counts)
print("Class weights:", class_weights.cpu().numpy())


# ============================================================
# 9. TRAINING AND VALIDATION FUNCTION
# ============================================================

def run_epoch(loader, training):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    #for images, geometry, labels in loader:
    for images, geometry, labels, motor_ids in loader:

        images = images.to(DEVICE)
        geometry = geometry.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            outputs = model(
                images,
                geometry
            )

            loss = criterion(
                outputs,
                labels
            )

            if training:
                loss.backward()
                optimizer.step()

        total_loss += (
            loss.item() * labels.size(0)
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


# ============================================================
# EVALUATE THE RESULT 
# ============================================================
def evaluate_motor_level(loader):

    model.eval()

    motor_predictions = {}
    motor_labels = {}

    with torch.no_grad():

        for images, geometry, labels, motor_ids in loader:

            images = images.to(DEVICE)
            geometry = geometry.to(DEVICE)

            outputs = model(images, geometry)

            probabilities = torch.softmax(outputs, dim=1)

            for i, unit in enumerate(motor_ids):

                unit = str(unit)

                if unit not in motor_predictions:
                    motor_predictions[unit] = []
                    motor_labels[unit] = labels[i].item()

                motor_predictions[unit].append(
                    probabilities[i].cpu().numpy()
                )

    correct = 0
    total = len(motor_predictions)

    print("\nMOTOR-LEVEL RESULTS")
    print("-------------------")

    for unit in sorted(motor_predictions):

        avg_prob = np.mean(
            motor_predictions[unit],
            axis=0
        )

        predicted_class = int(np.argmax(avg_prob))
        actual_class = motor_labels[unit]

        if predicted_class == actual_class:
            correct += 1

        print(
            f"{unit}: Actual={actual_class}, "
            f"Predicted={predicted_class}, "
            f"Probabilities={avg_prob.round(3)}"
        )

    accuracy = 100.0 * correct / total

    print(f"\nMotor-level accuracy: {accuracy:.2f}%")
    print(f"Correct motors: {correct}/{total}")

    return accuracy

# ============================================================
# 10. TRAINING LOOP
# ============================================================

BEST_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "best_motor_fusion.pth"
)

best_val_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(EPOCHS):

    train_loss, train_acc = run_epoch(
        train_loader,
        training=True
    )

    val_loss, val_acc = run_epoch(
        val_loader,
        training=False
    )

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_acc:.2f}%"
    )

    print(
        f"Val Loss:   {val_loss:.4f} | "
        f"Val Accuracy: {val_acc:.2f}%"
    )

    # ----------------------------------------
    # SAVE BEST VALIDATION MODEL
    # ----------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "geometry_mean": GEO_MEAN,
                "geometry_std": GEO_STD,
                "geometry_features": GEOMETRY_FEATURES,
                "class_map": CLASS_MAP,
                "image_width": IMAGE_W,
                "image_height": IMAGE_H
            },
            BEST_MODEL_PATH
        )

        print("  BEST MODEL SAVED")

    else:

        epochs_without_improvement += 1

    # ----------------------------------------
    # EARLY STOPPING
    # ----------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print(
            "\nEarly stopping: validation loss "
            "has not improved."
        )

        break


# ============================================================
# 11. LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("\nBest model loaded.")


# ============================================================
# 12. FINAL TEST EVALUATION
# ============================================================

model.eval()

test_loss, test_acc = run_epoch(
    test_loader,
    training=False
)

model.load_state_dict(
    torch.load(BEST_MODEL_PATH, map_location=DEVICE)
)

print("\n===================================")
print("FINAL TEST RESULTS")
print("===================================")

test_loss, test_acc = run_epoch(
    test_loader,
    training=False
)

print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_acc, 2), "%")

print("\nModel saved at:")
print(BEST_MODEL_PATH)

# Motor-level evaluation
evaluate_motor_level(test_loader)


