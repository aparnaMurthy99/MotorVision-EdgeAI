# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 08:22:30 2026

@author: HP
"""


from ultralytics import YOLO
import cv2
import os
import glob

# --------------------------------------------------
# 1. CROP FUNCTION — SAME 10% MARGIN
# --------------------------------------------------
def crop_with_margin(image, box, margin=0.10):

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    margin_x = margin * width
    margin_y = margin * height

    new_x1 = max(0, int(round(x1 - margin_x)))
    new_y1 = max(0, int(round(y1 - margin_y)))
    new_x2 = min(image.shape[1], int(round(x2 + margin_x)))
    new_y2 = min(image.shape[0], int(round(y2 + margin_y)))

    crop = image[new_y1:new_y2, new_x1:new_x2]

    return crop, (new_x1, new_y1, new_x2, new_y2)


# --------------------------------------------------
# 2. PATHS
# --------------------------------------------------
ROOT = r"D:\Competition_26_Code\Proj_5\BottomV_Model"

MODEL_PATH = os.path.join(
    ROOT, "runs", "detect", "BottomView", "weights", "best.pt"
)

DATASET_PATH = os.path.join(ROOT, "dataset", "images")

OUTPUT_ROOT = os.path.join(ROOT, "mobilenet_crops")

# Train and validation only. Existing test crops untouched.
SPLITS = ["train", "val"]

# --------------------------------------------------
# 3. LOAD YOLO MODEL
# --------------------------------------------------
model = YOLO(MODEL_PATH)

print("YOLO model loaded.")
print("Classes:", model.names)


# --------------------------------------------------
# 4. IMAGE EXTENSIONS
# --------------------------------------------------
extensions = (
    "*.jpg", "*.jpeg", "*.png",
    "*.bmp", "*.tif", "*.tiff"
)

total_images = 0
total_crops = 0


# --------------------------------------------------
# 5. PROCESS TRAIN AND VALIDATION
# --------------------------------------------------
for split in SPLITS:

    input_folder = os.path.join(DATASET_PATH, split)

    output_folder = os.path.join(OUTPUT_ROOT, split)

    os.makedirs(output_folder, exist_ok=True)

    image_paths = []

    for ext in extensions:
        image_paths.extend(
            glob.glob(os.path.join(input_folder, ext))
        )

    image_paths.sort()

    print("\n===================================")
    print("PROCESSING:", split.upper())
    print("Images:", len(image_paths))
    print("===================================")

    split_crops = 0

    for image_path in image_paths:

        filename = os.path.basename(image_path)
        image_name = os.path.splitext(filename)[0]

        image = cv2.imread(image_path)

        if image is None:
            print("ERROR reading:", filename)
            continue

        # Each motor/image gets its own folder
        image_crop_folder = os.path.join(
            output_folder, image_name
        )

        os.makedirs(image_crop_folder, exist_ok=True)

        results = model(image, conf=0.25, verbose=False)
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            print("No detections:", filename)
            continue

        # ------------------------------------------
        # IDENTIFY LARGEST FLANGE — CLASS ID 3
        # ------------------------------------------
        flange_indices = []

        for i, box in enumerate(result.boxes):

            cls = int(box.cls[0])

            if cls == 3:

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                area = (x2 - x1) * (y2 - y1)

                flange_indices.append((i, area))

        largest_flange_index = None

        if flange_indices:

            largest_flange_index = max(
                flange_indices,
                key=lambda item: item[1]
            )[0]

        # ------------------------------------------
        # CROP ALL DETECTED CLASSES
        # ------------------------------------------
        for box_index, box in enumerate(result.boxes):

            cls = int(box.cls[0])

            class_name = model.names[cls]

            # Keep only largest flange
            if cls == 3 and box_index != largest_flange_index:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            crop, coords = crop_with_margin(
                image,
                (x1, y1, x2, y2),
                margin=0.10
            )

            if crop.size == 0:
                print("Empty crop:", filename, class_name)
                continue

            crop_filename = f"{image_name}_{class_name}.jpg"

            crop_path = os.path.join(
                image_crop_folder,
                crop_filename
            )

            if cv2.imwrite(crop_path, crop):

                total_crops += 1
                split_crops += 1

                print(
                    split, "|", filename,
                    "|", class_name,
                    "|", crop.shape[1], "x", crop.shape[0]
                )

        total_images += 1

    print("\n", split.upper(), "COMPLETE")
    print("Images processed:", len(image_paths))
    print("Crops saved:", split_crops)


# --------------------------------------------------
# 6. SUMMARY
# --------------------------------------------------
print("\n===================================")
print("TRAIN + VALIDATION CROPPING DONE")
print("===================================")

print("Images processed:", total_images)
print("Total crops saved:", total_crops)

print("Output:", OUTPUT_ROOT)
