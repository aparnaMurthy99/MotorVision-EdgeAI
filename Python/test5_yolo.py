
from ultralytics import YOLO
import cv2
import os
import glob


# --------------------------------------------------
# 1. CROP FUNCTION — 10% MARGIN
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
# 2. LOAD TRAINED YOLO MODEL
# --------------------------------------------------
model = YOLO(
    r"D:\Competition_26_Code\Proj_5\BottomV_Model\runs\detect\BottomView\weights\best.pt"
)


# --------------------------------------------------
# 3. INPUT AND OUTPUT FOLDERS
# --------------------------------------------------
test_folder = (
    r"D:\Competition_26_Code\Proj_5\BottomV_Model\dataset\images\test"
)

annotated_folder = (
    r"D:\Competition_26_Code\Proj_5\BottomV_Model\YOLO_annotated_test"
)

crop_folder = (
    r"D:\Competition_26_Code\Proj_5\BottomV_Model\crops"
)

os.makedirs(annotated_folder, exist_ok=True)
os.makedirs(crop_folder, exist_ok=True)


# --------------------------------------------------
# 4. FIND ALL TEST IMAGES
# --------------------------------------------------
extensions = (
    "*.jpg", "*.jpeg", "*.png",
    "*.bmp", "*.tif", "*.tiff"
)

image_paths = []

for ext in extensions:
    image_paths.extend(
        glob.glob(os.path.join(test_folder, ext))
    )

image_paths.sort()

if not image_paths:
    print("No images found in:", test_folder)
    raise SystemExit

print("Total test images:", len(image_paths))


# --------------------------------------------------
# 5. PROCESS EVERY TEST IMAGE
# --------------------------------------------------
total_crops = 0
total_flange_crops = 0

for image_path in image_paths:

    filename = os.path.basename(image_path)
    image_name = os.path.splitext(filename)[0]

    print("\n======================================")
    print("Processing image:", filename)
    print("======================================")

    image = cv2.imread(image_path)

    if image is None:
        print("ERROR: Could not read image:", image_path)
        continue

    image_crop_folder = os.path.join(crop_folder, image_name)
    os.makedirs(image_crop_folder, exist_ok=True)

    results = model(image, conf=0.25, verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:

        print("No objects detected.")

    else:

        # ------------------------------------------
        # 1. PRINT ALL DETECTIONS
        # ------------------------------------------
        print("\nDetected objects:")

        for i, box in enumerate(result.boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            print(
                f"Index: {i} | "
                f"Class ID: {cls} | "
                f"Name: {model.names[cls]} | "
                f"Confidence: {conf:.4f}"
            )

        # ------------------------------------------
        # 2. IDENTIFY LARGEST FLANGE BOX
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

            print(
                "\nLargest flange selected:",
                largest_flange_index
            )

        else:
            print("\nWARNING: No flangeFace detected (class ID 3).")

        # ------------------------------------------
        # 3. PROCESS DETECTIONS
        # ------------------------------------------
        for box_index, box in enumerate(result.boxes):

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = model.names[cls]

            # Keep ONLY the largest flange bounding box
            if cls == 3 and box_index != largest_flange_index:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            print(
                "\nCropping:",
                class_name,
                "| Confidence:", round(conf, 4),
                "| Box index:", box_index
            )

            crop, coords = crop_with_margin(
                image,
                (x1, y1, x2, y2),
                margin=0.10
            )

            if crop.size == 0:
                print("WARNING: Empty crop; skipped.")
                continue

            # --------------------------------------
            # SAVE CROP
            # --------------------------------------
            crop_filename = (
                f"{image_name}_{class_name}.jpg"
            )

            crop_path = os.path.join(
                image_crop_folder,
                crop_filename
            )

            saved = cv2.imwrite(crop_path, crop)

            if saved:

                total_crops += 1

                if cls == 3:
                    total_flange_crops += 1

                print("Crop saved:", crop_path)

                print(
                    "Crop dimensions:",
                    crop.shape[1], "x", crop.shape[0]
                )

                print("Crop coordinates:", coords)

            else:
                print("ERROR saving crop:", crop_path)

    # ----------------------------------------------
    # 4. SAVE ANNOTATED IMAGE
    # ----------------------------------------------
    annotated_image = result.plot()

    annotated_path = os.path.join(
        annotated_folder,
        filename
    )

    cv2.imwrite(annotated_path, annotated_image)

    print("\nAnnotated image saved:", annotated_path)


# --------------------------------------------------
# 6. SUMMARY
# --------------------------------------------------
print("\n======================================")
print("TESTING AND CROPPING COMPLETE")
print("======================================")

print("Test images processed:", len(image_paths))
print("Total crops saved:", total_crops)
print("Flange crops saved:", total_flange_crops)

print("Crops folder:", crop_folder)
print("Annotated images folder:", annotated_folder)

print("======================================")
