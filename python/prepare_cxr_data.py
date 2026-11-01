#!/usr/bin/env python3
"""
CXR Dataset Preparation
Downloads and prepares CheXpert (classification) and Montgomery/JSRT (segmentation)
into the format expected by train_cxr.py
"""

import os
import json
import argparse
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests

CHEXNET_PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]

NIH_PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]

# NIH ChestX-ray14 mapping (same 14 classes)
NIH_TO_CHEXNET = {n: n for n in NIH_PATHOLOGIES}

# CheXpert pathologies (includes more, we map to 14)
CHEXPERT_PATHOLOGIES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

CHEXPERT_TO_CHEXNET = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Pleural Effusion": "Effusion",
    "Lung Opacity": "Infiltration",  # approximate
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    "Pleural Other": "Pleural_Thickening",
    "Lung Lesion": "Mass",
    "Enlarged Cardiomediastinum": None,
    "No Finding": None,
    "Fracture": None,
    "Support Devices": None,
}


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Download with progress bar."""
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=dest.name) as pbar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def prepare_chexpert(data_dir: Path, chexpert_root: Path) -> Dict[str, List[int]]:
    """
    Process CheXpert CSV into labels.json format.
    Expects chexpert_root/ with train.csv, valid.csv and images/
    """
    print("Processing CheXpert...")
    labels = {}

    for split in ["train", "valid"]:
        csv_path = chexpert_root / f"{split}.csv"
        if not csv_path.exists():
            print(f"  {csv_path} not found, skipping")
            continue

        df = pd.read_csv(csv_path)
        print(f"  {split}: {len(df)} rows")

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split}"):
            # Path relative to CheXpert root
            rel_path = row["Path"]
            # CheXpert paths like "train/patient123/study456/image.png"
            # We'll copy to flat images/ dir with unique names
            src = chexpert_root / rel_path
            if not src.exists():
                continue

            # Build unique filename
            parts = Path(rel_path).parts
            fname = "_".join(parts[1:])  # skip "train"/"valid"
            dst = data_dir / "images" / fname

            # Copy if not already there
            if not dst.exists():
                shutil.copy2(src, dst)

            # Map pathologies
            path_labels = [0] * len(CHEXNET_PATHOLOGIES)
            for chexpert_name, chexnet_name in CHEXPERT_TO_CHEXNET.items():
                if chexnet_name is None:
                    continue
                if chexpert_name in df.columns:
                    val = row[chexpert_name]
                    # CheXpert: 1.0 = positive, -1.0 = uncertain, 0.0 = negative, NaN = not mentioned
                    if val == 1.0 or val == -1.0:  # treat uncertain as positive
                        idx = CHEXNET_PATHOLOGIES.index(chexnet_name)
                        path_labels[idx] = 1

            labels[fname] = {"pathologies": path_labels}

    return labels


def prepare_nih(data_dir: Path, nih_root: Path) -> Dict[str, List[int]]:
    """
    Process NIH ChestX-ray14 Data_Entry_2017.csv into labels.json
    """
    print("Processing NIH ChestX-ray14...")
    labels = {}

    csv_path = nih_root / "Data_Entry_2017.csv"
    if not csv_path.exists():
        print(f"  {csv_path} not found")
        return labels

    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows")

    images_dir = nih_root / "images"
    if not images_dir.exists():
        # Check for images_001 etc.
        for i in range(1, 13):
            d = nih_root / f"images_{i:03d}"
            if d.exists():
                images_dir = d
                break

    if not images_dir.exists():
        print("  No images directory found")
        return labels

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  NIH"):
        fname = row["Image Index"]
        src = images_dir / fname
        if not src.exists():
            continue

        dst = data_dir / "images" / fname
        if not dst.exists():
            shutil.copy2(src, dst)

        # Finding Labels: "Atelectasis|Cardiomegaly|..."
        finding_str = row["Finding Labels"]
        path_labels = [0] * len(CHEXNET_PATHOLOGIES)

        for finding in finding_str.split("|"):
            if finding in NIH_TO_CHEXNET:
                idx = CHEXNET_PATHOLOGIES.index(NIH_TO_CHEXNET[finding])
                path_labels[idx] = 1

        labels[fname] = {"pathologies": path_labels}

    return labels


def prepare_segmentation_montgomery(data_dir: Path, montgomery_root: Path):
    """Process Montgomery County X-ray set (masks for lungs)."""
    print("Processing Montgomery segmentation masks...")
    seg_dir = data_dir / "segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Montgomery has: CXR PNGs + manual masks (left lung, right lung, heart)
    # Masks are in Montgomery/ManualMask/leftLung, rightLung, heart
    mask_dirs = {
        "leftLung": 1,
        "rightLung": 2,
        "heart": 3,
    }

    cxr_dir = montgomery_root / "CXR_png"
    if not cxr_dir.exists():
        print("  CXR_png not found")
        return

    for cxr_file in tqdm(list(cxr_file.glob("*.png")), desc="  Montgomery"):
        stem = cxr_file.stem  # e.g., "MCUC_Xray_001"
        # Combine masks into single 5-class mask
        combined = np.zeros((1024, 1024), dtype=np.uint8)  # Montgomery is 4020x4892? resize later

        for mask_name, class_idx in mask_dirs.items():
            mask_file = montgomery_root / "ManualMask" / mask_name / f"{stem}.png"
            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                    combined[mask > 128] = class_idx

        # Save combined mask
        out_path = seg_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), combined)


def prepare_segmentation_jsrt(data_dir: Path, jsrt_root: Path):
    """Process JSRT dataset (has lung/heart masks)."""
    print("Processing JSRT segmentation masks...")
    seg_dir = data_dir / "segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # JSRT structure: images/ + masks/ (lung masks, heart masks)
    # This is a placeholder - actual JSRT structure varies
    pass


def main():
    parser = argparse.ArgumentParser(description="Prepare CXR datasets for training")
    parser.add_argument("--output-dir", type=str, default="data/cxr", help="Output directory")
    parser.add_argument("--chexpert", type=str, help="Path to CheXpert root (with train.csv, valid.csv, images/)")
    parser.add_argument("--nih", type=str, help="Path to NIH ChestX-ray14 root")
    parser.add_argument("--montgomery", type=str, help="Path to Montgomery root")
    parser.add_argument("--jsrt", type=str, help="Path to JSRT root")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    all_labels = {}

    if args.chexpert:
        all_labels.update(prepare_chexpert(out_dir, Path(args.chexpert)))

    if args.nih:
        all_labels.update(prepare_nih(out_dir, Path(args.nih)))

    # Add empty seg_mask field for classification-only images
    for fname in all_labels:
        all_labels[fname]["seg_mask"] = None

    # Write labels.json
    labels_path = out_dir / "labels.json"
    with open(labels_path, 'w') as f:
        json.dump(all_labels, f, indent=2)

    print(f"\nWrote {len(all_labels)} labels to {labels_path}")

    if args.montgomery:
        prepare_segmentation_montgomery(out_dir, Path(args.montgomery))
        # Update labels.json with seg_mask paths
        # (Montgomery images need to be copied to images/ too)
        # For now, just note the masks exist
        print("Masks written to segmentation/")

    if args.jsrt:
        prepare_segmentation_jsrt(out_dir, Path(args.jsrt))


if __name__ == "__main__":
    main()