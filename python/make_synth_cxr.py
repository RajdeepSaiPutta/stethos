#!/usr/bin/env python3
"""
Quick synthetic data test for CXR training pipeline.
Creates fake data matching the expected format to verify the training script runs.
"""

import json
import numpy as np
import cv2
from pathlib import Path
import torch

def create_synthetic_dataset(out_dir: Path, n_samples: int = 64, img_size: int = 256):
    """Create synthetic CXR-like data for pipeline testing."""
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    seg_dir = out_dir / "segmentation"
    img_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    labels = {}

    for i in range(n_samples):
        fname = f"synth_{i:04d}.png"

        # Synthetic "X-ray" - noise + some structure
        img = np.random.randn(img_size, img_size).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        # Add some "lung-like" ellipses
        cv2.ellipse(img, (img_size//3, img_size//2), (img_size//4, img_size//3), 0, 0, 360, 0.8, -1)
        cv2.ellipse(img, (2*img_size//3, img_size//2), (img_size//4, img_size//3), 0, 0, 360, 0.8, -1)
        # Heart-ish
        cv2.ellipse(img, (img_size//2, 2*img_size//3), (img_size//6, img_size//5), 0, 0, 360, 0.6, -1)

        img_uint8 = (img * 255).astype(np.uint8)
        cv2.imwrite(str(img_dir / fname), img_uint8)

        # Random pathology labels (multi-label)
        path_labels = np.random.binomial(1, 0.15, size=14).tolist()

        # Synthetic segmentation mask (5 classes)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        # background = 0
        cv2.ellipse(mask, (img_size//3, img_size//2), (img_size//4, img_size//3), 0, 0, 360, 1, -1)   # left lung
        cv2.ellipse(mask, (2*img_size//3, img_size//2), (img_size//4, img_size//3), 0, 0, 360, 2, -1)  # right lung
        cv2.ellipse(mask, (img_size//2, 2*img_size//3), (img_size//6, img_size//5), 0, 0, 360, 3, -1)  # heart
        # trachea - thin vertical
        cv2.rectangle(mask, (img_size//2-2, img_size//6), (img_size//2+2, img_size//2), 4, -1)

        cv2.imwrite(str(seg_dir / fname), mask)

        labels[fname] = {
            "pathologies": path_labels,
            "seg_mask": fname
        }

    with open(out_dir / "labels.json", 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"Created {n_samples} synthetic samples in {out_dir}")
    print(f"  images/: {len(list(img_dir.glob('*.png')))}")
    print(f"  segmentation/: {len(list(seg_dir.glob('*.png')))}")
    print(f"  labels.json: {len(labels)} entries")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="data/cxr_synth")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=256)
    args = parser.parse_args()

    create_synthetic_dataset(Path(args.out_dir), args.n_samples, args.img_size)