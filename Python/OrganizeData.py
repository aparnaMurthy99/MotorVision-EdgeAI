# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 09:02:24 2026

@author: HP
"""

import os
import glob
import shutil
import cv2
import pandas as pd

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = r"D:\Competition_26_Code\Proj_5\BottomV_Model"

SPLITS = {
    "train": {
        "crops": os.path.join(ROOT, "mobilenet_crops", "train"),
        "csv": os.path.join(ROOT, "GeometricalFeats_train.csv")
    },
    "val": {
        "crops": os.path.join(ROOT, "mobilenet_crops", "val"),
        "csv": os.path.join(ROOT, "GeometricalFeats_val.csv")
    },
    "test": {
        "crops": os.path.join(ROOT, "crops"),
        "csv": os.path.join(ROOT, "GeometricalFeats_test.csv")
    }
}

OUTPUT_ROOT = os.path.join(ROOT, "MobileNetDataset")

CLASSES = [
    "gearTooth",
    "gearFace",
    "mainFace",
    "flangeFace"
]

EXPECTED_SHAPE = (384, 480, 3)

# --------------------------------------------------
# 2. CLEAR OLD OUTPUT ONLY
# --------------------------------------------------

if os.path.exists(OUTPUT_ROOT):
    shutil.rmtree(OUTPUT_ROOT)

# --------------------------------------------------
# 3. CREATE CLASS FOLDERS
# --------------------------------------------------

for split in SPLITS:
    for label in ["NO_FAULT", "FAULT"]:
        os.makedirs(
            os.path.join(OUTPUT_ROOT, split, label),
            exist_ok=True
        )

# --------------------------------------------------
# 4. COPY CROPS ACCORDING TO MOTOR LABEL
# --------------------------------------------------

for split, paths in SPLITS.items():

    CROPS_ROOT = paths["crops"]
    CSV_PATH = paths["csv"]

    df = pd.read_csv(CSV_PATH)

    label_map = dict(
        zip(df["Unit"].astype(str), df["LABEL"].astype(int))
    )

    copied = 0
    missing = []
    wrong_size = []

    for folder in sorted(os.listdir(CROPS_ROOT)):

        motor_path = os.path.join(CROPS_ROOT, folder)

        if not os.path.isdir(motor_path):
            continue

        if folder not in label_map:
            print("WARNING: Missing from CSV:", folder)
            missing.append(folder)
            continue

        label = label_map[folder]

        class_name = "FAULT" if label == 1 else "NO_FAULT"

        crop_files = []

        for class_type in CLASSES:
            matches = glob.glob(
                os.path.join(motor_path, f"*{class_type}*.jpg")
            )
            crop_files.extend(matches)

        if len(crop_files) != 4:
            print(
                f"WARNING: {folder} has {len(crop_files)} crops, expected 4"
            )
            continue

        # Verify all four images before copying
        valid = True

        for src in crop_files:

            img = cv2.imread(src)

            if img is None or img.shape != EXPECTED_SHAPE:
                print(
                    "INCORRECT SIZE:",
                    folder,
                    os.path.basename(src),
                    None if img is None else img.shape
                )
                wrong_size.append(src)
                valid = False

        if not valid:
            continue

        for src in crop_files:

            filename = os.path.basename(src)

            dst = os.path.join(
                OUTPUT_ROOT,
                split,
                class_name,
                f"{folder}_{filename}"
            )

            shutil.copy2(src, dst)
            copied += 1

    print(f"\n{split.upper()}: {copied} crops copied")
    print("Missing CSV entries:", missing)
    print("Incorrect-size images:", len(wrong_size))

print("\nMobileNetV3 dataset preparation complete!")


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

# ROOT = r"D:\Competition_26_Code\Proj_5\BottomV_Model"

# SPLITS = {
#     "train": {
#         "crops": os.path.join(ROOT, "mobilenet_crops", "train"),
#         "csv": os.path.join(ROOT, "GeometricalFeats_train.csv")
#     },
#     "val": {
#         "crops": os.path.join(ROOT, "mobilenet_crops", "val"),
#         "csv": os.path.join(ROOT, "GeometricalFeats_val.csv")
#     },
#     "test": {
#         "crops": os.path.join(ROOT, "crops"),
#         "csv": os.path.join(ROOT, "GeometricalFeats_test.csv")
#     }
# }

# OUTPUT_ROOT = os.path.join(ROOT, "MobileNetDataset")

# # --------------------------------------------------
# # 2. CREATE CLASS FOLDERS
# # --------------------------------------------------

# for split in SPLITS:
#     for label in ["NO_FAULT", "FAULT"]:
#         os.makedirs(
#             os.path.join(OUTPUT_ROOT, split, label),
#             exist_ok=True
#         )

# # --------------------------------------------------
# # 3. COPY CROPS ACCORDING TO MOTOR LABEL
# # --------------------------------------------------

# for split, paths in SPLITS.items():

#     CROPS_ROOT = paths["crops"]
#     CSV_PATH = paths["csv"]

#     df = pd.read_csv(CSV_PATH)

#     # Match Unit names to their labels
#     label_map = dict(zip(df["Unit"].astype(str), df["LABEL"]))

#     copied = 0
#     missing = []

#     for folder in sorted(os.listdir(CROPS_ROOT)):

#         motor_path = os.path.join(CROPS_ROOT, folder)

#         if not os.path.isdir(motor_path):
#             continue

#         if folder not in label_map:
#             print(f"WARNING: {folder} missing from CSV")
#             missing.append(folder)
#             continue

#         label = int(label_map[folder])

#         class_name = "FAULT" if label == 1 else "NO_FAULT"

#         crop_files = []

#         for class_type in [
#             "gearTooth",
#             "gearFace",
#             "mainFace",
#             "flangeFace"
#         ]:
#             crop_files.extend(
#                 glob.glob(
#                     os.path.join(motor_path, f"*{class_type}*.jpg")
#                 )
#             )

#         if len(crop_files) != 4:
#             print(
#                 f"WARNING: {folder} has {len(crop_files)} crops, expected 4"
#             )
#             continue

#         for src in crop_files:

#             filename = os.path.basename(src)

#             dst = os.path.join(
#                 OUTPUT_ROOT,
#                 split,
#                 class_name,
#                 f"{folder}_{filename}"
#             )

#             shutil.copy2(src, dst)
#             copied += 1

#     print(f"\n{split.upper()}: {copied} crops copied")
#     print("Missing CSV entries:", missing)

# print("\nMobileNetV3 dataset preparation complete!")