# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 20:48:16 2026

@author: HP
"""

import cv2
import numpy as np
import pandas as pd
import os, re, glob
from scipy.ndimage import map_coordinates, gaussian_filter1d

def refine_circle(gray, cx, cy, r, band=0.08, n=360):
    g = gray.astype(np.float32)
    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    half = max(6.0, band * r)
    off = np.arange(-half, half + 0.5, 0.5)           # sample every 0.5 px along each ray

    rr = r + off[None, :]
    xs = cx + np.cos(ang)[:, None] * rr
    ys = cy + np.sin(ang)[:, None] * rr
    prof = map_coordinates(g, [ys, xs], order=1, mode="nearest")   # (n, m)
    grad = np.gradient(gaussian_filter1d(prof, 1.0, axis=1), axis=1)
    k = np.abs(grad).argmax(axis=1)                    # strongest edge on each ray
    pts = np.c_[xs[np.arange(n), k], ys[np.arange(n), k]]

    for _ in range(4):                                 # circle fit + outlier rejection
        A = np.c_[2*pts[:, 0], 2*pts[:, 1], np.ones(len(pts))]
        b = (pts**2).sum(1)
        c0, c1, c2 = np.linalg.lstsq(A, b, rcond=None)[0]
        cx, cy = c0, c1
        r = np.sqrt(c2 + cx**2 + cy**2)
        d = np.abs(np.hypot(pts[:, 0]-cx, pts[:, 1]-cy) - r)
        keep = d < max(3*np.median(d), 0.75)
        if keep.sum() < 30:
            break
        pts = pts[keep]
    return cx, cy, r

def count_teeth(image_path):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    a = np.linspace(0, 2*np.pi, 720, endpoint=False)
    best = 0, 0
    for r in np.arange(0.5, 0.85, 0.02) * min(h, w) / 2:   # try a few circles
        v = map_coordinates(gray.astype(np.float32),
                            [h/2 + r*np.sin(a), w/2 + r*np.cos(a)], order=1)
        s = np.abs(np.fft.rfft(v - v.mean()))[6:40]
        if s.max() > best[0]:
            best = s.max(), s.argmax() + 6
    return best[1]

def OD_Calculate(Path, Choice):

    # Select reference calibration
    if Choice == 1:
        ref_od_px = OD_in_px
        mm_per_px = MM_PER_PX

    elif Choice == 2:
        ref_od_px = OD_in_pxG
        mm_per_px = OD_computed / OD_in_pxG

    else:
        raise ValueError("Choice must be 1 (Flange) or 2 (Gear)")

    # 1. Load test image
    img = cv2.imread(Path)

    if img is None:
        raise FileNotFoundError(Path)

    h, w = img.shape[:2]

    # 2. Detect centre and radius using bounded HoughCircles
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)

    DETECTION_SCALE = 1000 / max(h, w)

    small_gray = cv2.resize(
        blur,
        None,
        fx=DETECTION_SCALE,
        fy=DETECTION_SCALE,
        interpolation=cv2.INTER_AREA
    )

    expected_radius_small = (ref_od_px / 2) * DETECTION_SCALE

    min_rad = int(expected_radius_small * 0.97)
    max_rad = int(expected_radius_small * 1.03)

    circle = None

    for p2 in (50, 40, 30, 20):

        c = cv2.HoughCircles(
            small_gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=small_gray.shape[0],
            param1=100,
            param2=p2,
            minRadius=min_rad,
            maxRadius=max_rad
        )

        if c is not None:
            circle = c[0][0]
            break

    if circle is None:
        raise RuntimeError(f"Circle detection failed: {Path}")

    cx_small, cy_small, r_small = circle

    cx = cx_small / DETECTION_SCALE
    cy = cy_small / DETECTION_SCALE
    outer_radius = r_small / DETECTION_SCALE
    
    cx, cy, outer_radius = refine_circle(gray, cx, cy, outer_radius)

    # 3. Calculate OD
    od_mm = (outer_radius * 2) * mm_per_px

    # 4. Flange: calculate Laplacian variance
    if Choice == 1:

        ring_px = RING_MM / mm_per_px
        inner_radius = int(round(
            outer_radius - ring_px - EXTRA_PX
        ))

        if inner_radius <= 0:
            raise RuntimeError("Invalid inner ring radius")

        outer_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(
            outer_mask,
            (int(cx), int(cy)),
            int(outer_radius),
            255,
            -1
        )

        inner_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(
            inner_mask,
            (int(cx), int(cy)),
            inner_radius,
            255,
            -1
        )

        ring_mask = cv2.subtract(outer_mask, inner_mask)

        kernel = np.ones((7, 7), np.uint8)
        safe_ring_mask = cv2.erode(
            ring_mask, kernel, iterations=2
        )

        laplacian = cv2.Laplacian(blur, cv2.CV_64F)
        laplacian_vals = laplacian[safe_ring_mask > 0]

        if len(laplacian_vals) == 0:
            raise RuntimeError("Empty flange measurement region")

        laplacian_var = float(np.var(laplacian_vals))

        return od_mm, laplacian_var

    # 5. Gear: return only OD
    else:
        return od_mm

def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def extract_geometry(CROPS_ROOT, OUT_CSV):
    rows = []

    for folder in sorted(os.listdir(CROPS_ROOT), key=natural_key):
        d = os.path.join(CROPS_ROOT, folder)
        if not os.path.isdir(d):
            continue
    
        flange_files = glob.glob(os.path.join(d, "*flangeFace*.jpg"))
        gear_files   = glob.glob(os.path.join(d, "*gearFace*.jpg"))
    
        row = {"Unit": folder, "OD_Flange": None, "OD_Gear": None,
           "GearCount": None, "Lap_Var": None, "BlowHoles": None,
           "LABEL":None}
        
        # Flange: OD + blow-hole metric (Laplacian variance)
        if flange_files:
            try:
                od, lap_var = OD_Calculate(flange_files[0], 1)
                row["OD_Flange"] = round(od, 2)
                row["Lap_Var"]   = round(lap_var, 2)
                row["BlowHoles"] = 1 if lap_var > 10 else 0
            except Exception as e:
                print(f"[{folder}] flange failed: {e}")
    
        # Gear: OD + tooth count
        if gear_files:
            try:
                row["OD_Gear"]   = round(OD_Calculate(gear_files[0], 2), 2)
                row["GearCount"] = count_teeth(gear_files[0])
            except Exception as e:
                print(f"[{folder}] gear failed: {e}")
                
        if (
            row["OD_Flange"] < 99.0 or row["OD_Flange"] > 101.0
            or row["OD_Gear"] < 21.0 or row["OD_Gear"] > 22.5
            or row["GearCount"] != 15
            or row["BlowHoles"] == 1
        ):
            row["LABEL"] = 1       # FAULT
        else:
            row["LABEL"] = 0       # NO_FAULT
    
        rows.append(row)
        print(row)
    
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(rows)} rows to {OUT_CSV}")
    
    
# # --- Configuration ---
# FLANGE_PATH = r"D:\Competition_26_Code\Proj_5\BottomV_Model\crops\bv8\bv8_flangeFace.jpg"
# Flcsv_PATH = r"D:\Competition_26_Code\GeometricalFeatures\FlangeGeo.csv"

# GEAR_PATH = r"D:\Competition_26_Code\Proj_5\BottomV_Model\crops\bv36\bv36_gearFace.jpg"
# Gecsv_PATH = r"D:\Competition_26_Code\GeometricalFeatures\GearGeo.csv"

# RING_MM = 5.0
# EXTRA_PX = 5


# # 1. Load flange reference geometry
# ref = pd.read_csv(Flcsv_PATH)

# MM_PER_PX = float(ref.loc[0, "MM_per_px"])
# OD_in_px = float(ref.loc[0, "OD_in_px"])
# FlcenterX = float(ref.loc[0, "CenterX"])
# FlcenterY = float(ref.loc[0, "CenterY"])


# # 2. Load gear reference geometry
# ref2 = pd.read_csv(Gecsv_PATH)

# OD_computed = float(ref2.loc[0, "OD_computed"])
# OD_in_pxG = float(ref2.loc[0, "OD_in_px"])
# GrcenterX = float(ref2.loc[0, "CenterX"])
# GrcenterY = float(ref2.loc[0, "CenterY"])


# # 3. Display reference values
# print("--------------- FLANGE REFERENCE ----------------")
# print(f"Center: ({FlcenterX:.2f}, {FlcenterY:.2f})")
# print(f"Reference OD: {OD_in_px:.2f} px")
# print(f"MM_PER_PX: {MM_PER_PX:.6f}")

# print("--------------- GEAR REFERENCE ------------------")
# print(f"Center: ({GrcenterX:.2f}, {GrcenterY:.2f})")
# print(f"Reference OD: {OD_in_pxG:.2f} px")
# print(f"Reference OD: {OD_computed:.2f} mm")

# # Flange
# Flange_OD, Flange_Variance = OD_Calculate(FLANGE_PATH, 1)

# # Gear
# Gear_OD = OD_Calculate(GEAR_PATH, 2)

# print(f"Flange OD: {Flange_OD:.2f} mm")
# print(f"Flange Laplacian Variance: {Flange_Variance:.2f}")
# print(f"Gear OD: {Gear_OD:.2f} mm")

# TeethCnt = count_teeth(GEAR_PATH)
# print("Gear-Count : ",TeethCnt)


# ---------- 1. Configuration ----------
Flcsv_PATH = r"D:\Competition_26_Code\GeometricalFeatures\FlangeGeo.csv"
Gecsv_PATH = r"D:\Competition_26_Code\GeometricalFeatures\GearGeo.csv"

ROOT = r"D:\Competition_26_Code\Proj_5\BottomV_Model"

TRAIN_CROP_PATH = os.path.join(ROOT, "mobilenet_crops", "train")
VAL_CROP_PATH   = os.path.join(ROOT, "mobilenet_crops", "val")

OUT_CSV1 = r"D:\Competition_26_Code\GeometricalFeatures\GeometricalFeats_train.csv"
OUT_CSV2 = r"D:\Competition_26_Code\GeometricalFeatures\GeometricalFeats_val.csv"

RING_MM  = 5.0
EXTRA_PX = 5

# ---------- 2. Load reference calibration ONCE ----------
ref = pd.read_csv(Flcsv_PATH)
MM_PER_PX = float(ref.loc[0, "MM_per_px"])
OD_in_px  = float(ref.loc[0, "OD_in_px"])
FlcenterX = float(ref.loc[0, "CenterX"])
FlcenterY = float(ref.loc[0, "CenterY"])

ref2 = pd.read_csv(Gecsv_PATH)
OD_computed = float(ref2.loc[0, "OD_computed"])
OD_in_pxG   = float(ref2.loc[0, "OD_in_px"])
GrcenterX   = float(ref2.loc[0, "CenterX"])
GrcenterY   = float(ref2.loc[0, "CenterY"])

extract_geometry(TRAIN_CROP_PATH, OUT_CSV1)
extract_geometry(VAL_CROP_PATH, OUT_CSV2)
# imgPth = r"D:\Competition_26_Code\Proj_5\BottomV_Model\crops\bv35\bv35_flangeFace.jpg"
# gray = cv2.imread(imgPth, cv2.IMREAD_GRAYSCALE)
# cx, cy, r = refine_circle(gray, FlcenterX, FlcenterY, OD_in_px / 2)
# print("new center values : " , int(cx), int(cy))
# print("new center values int : " , cx, cy)
# print("OD px (refined):", 2 * r, "  CSV:", OD_in_px)
# print("OD mm (refined):", 2 * r * MM_PER_PX, "  expected: 100.00")

# vis = cv2.imread(imgPth)
# cv2.circle(vis, (int(FlcenterX), int(FlcenterY)), int(OD_in_px/2), (0, 0, 255), 1)   # MATLAB (red)
# cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 0), 1)                          # refined (green)
# cv2.imwrite("ref_check.png", vis)







