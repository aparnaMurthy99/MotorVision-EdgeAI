# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:43:34 2026

@author: HP
"""


import cv2
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(r"D:\Competition_26_Code\Proj_5\BottomV_Model")

IMAGE_ROOT = ROOT / "mobilenetDataset" / "train"

GEOMETRY_CSV = ROOT / "GeometricalFeats_train.csv"

OUTPUT_DIR = (
    ROOT / "MobileNetV3_Results" / "TF_SavedModel"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_OUTPUT = OUTPUT_DIR / "calibration_images.npy"
GEOMETRY_OUTPUT = OUTPUT_DIR / "calibration_geometry.npy"

# --------------------------------------------------
# 2. READ GEOMETRY DATA
# --------------------------------------------------

print("Loading geometry data...")

geo_data = np.genfromtxt(
    GEOMETRY_CSV,
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8-sig"
)

geo_data = np.atleast_1d(geo_data)

geometry_lookup = {}

for row in geo_data:

    unit = str(row["Unit"]).strip()

    geometry_lookup[unit] = [
        float(row["OD_Flange"]),
        float(row["OD_Gear"]),
        float(row["GearCount"])
    ]

print("Geometry records:", len(geometry_lookup))

# --------------------------------------------------
# 3. COLLECT ALL TRAINING IMAGES
# --------------------------------------------------

records = []

for class_name in ["FAULT", "NO_FAULT"]:

    folder = IMAGE_ROOT / class_name

    if not folder.exists():
        raise FileNotFoundError(folder)

    files = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    print(class_name, "images:", len(files))

    for image_path in files:

        unit = image_path.name.split("_")[0]

        if unit not in geometry_lookup:
            raise ValueError(
                f"Missing geometry for {image_path.name}"
            )

        records.append(
            (image_path, unit, geometry_lookup[unit])
        )

N = len(records)

print("\nTotal calibration images:", N)

if N == 0:
    raise ValueError("No training images found.")

# --------------------------------------------------
# 4. CREATE IMAGE ARRAY
# --------------------------------------------------

# Shape: N x Height x Width x Channels
# Each image: 384 x 480 x 3
# Dtype: float32
# Values: 0 to 1

print("\nCreating image calibration array...")

cal_images = np.lib.format.open_memmap(
    IMAGE_OUTPUT,
    mode="w+",
    dtype=np.float32,
    shape=(N, 384, 480, 3)
)

cal_geometry = np.zeros(
    (N, 3),
    dtype=np.float32
)

for i, (image_path, unit, geo_values) in enumerate(records):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Cannot read image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # No resizing: training images were already 480 x 384.
    if image.shape != (384, 480, 3):
        raise ValueError(
            f"Incorrect image dimensions: "
            f"{image_path.name}, {image.shape}"
        )

    # Convert uint8 [0,255] to float32 [0,1].
    # Do NOT apply ImageNet normalization here.
    image = image.astype(np.float32) / 255.0

    cal_images[i] = image

    # Raw geometry values; ONNX performs normalization.
    cal_geometry[i] = geo_values

    if (i + 1) % 40 == 0:
        print(f"Processed {i + 1}/{N}")

# Flush image data to disk.
cal_images.flush()

del cal_images

# --------------------------------------------------
# 5. SAVE GEOMETRY ARRAY
# --------------------------------------------------

np.save(
    GEOMETRY_OUTPUT,
    cal_geometry
)

# --------------------------------------------------
# 6. VERIFY SAVED ARRAYS
# --------------------------------------------------

print("\nVerifying saved calibration arrays...")

images_check = np.load(
    IMAGE_OUTPUT,
    mmap_mode="r"
)

geometry_check = np.load(
    GEOMETRY_OUTPUT
)

print("\nIMAGE ARRAY")
print("Shape:", images_check.shape)
print("Dtype:", images_check.dtype)
print("Minimum:", images_check.min())
print("Maximum:", images_check.max())

print("\nGEOMETRY ARRAY")
print("Shape:", geometry_check.shape)
print("Dtype:", geometry_check.dtype)
print("First row:", geometry_check[0])

print("\nCalibration arrays saved successfully.")
print("Image file:", IMAGE_OUTPUT)
print("Geometry file:", GEOMETRY_OUTPUT)
