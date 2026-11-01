#!/usr/bin/env python3
"""
Synthetic spirometry data generator for testing.
Creates patients.json with longitudinal PFT data and pattern labels.
"""

import json
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from typing import Dict

PFT_PARAMETERS = [
    "FVC", "FEV1", "FEV1_FVC", "FEF25_75", "PEF",
    "FIVC", "FIF50", "MVV", "TLC", "RV", "DLCO"
]

PATTERN_CLASSES = ["normal", "obstructive", "restrictive", "mixed"]


def generate_patient(patient_id: str, pattern: int, n_visits: int = 5) -> Dict:
    """Generate synthetic longitudinal PFT data for a patient with a given pattern."""
    base_date = datetime(2020, 1, 1)
    visits = []

    # Base values (roughly normal adult male)
    base = {
        "FVC": 4.5,
        "FEV1": 3.6,
        "FEV1_FVC": 0.80,
        "FEF25_75": 3.5,
        "PEF": 9.0,
        "FIVC": 4.5,
        "FIF50": 5.0,
        "MVV": 140,
        "TLC": 6.0,
        "RV": 1.5,
        "DLCO": 25,
    }

    if pattern == 0:  # normal
        params_fn = lambda v: {k: v + np.random.normal(0, v*0.05) for k, v in base.items()}
    elif pattern == 1:  # obstructive
        params_fn = lambda v: {
            "FVC": v["FVC"] * 0.9 + np.random.normal(0, 0.1),
            "FEV1": v["FEV1"] * 0.6 + np.random.normal(0, 0.1),
            "FEV1_FVC": v["FEV1_FVC"] * 0.7 + np.random.normal(0, 0.02),
            "FEF25_75": v["FEF25_75"] * 0.4 + np.random.normal(0, 0.2),
            "PEF": v["PEF"] * 0.7 + np.random.normal(0, 0.3),
            "FIVC": v["FIVC"] * 0.95 + np.random.normal(0, 0.1),
            "FIF50": v["FIF50"] * 0.85 + np.random.normal(0, 0.2),
            "MVV": v["MVV"] * 0.7 + np.random.normal(0, 5),
            "TLC": v["TLC"] * 1.1 + np.random.normal(0, 0.2),
            "RV": v["RV"] * 1.4 + np.random.normal(0, 0.2),
            "DLCO": v["DLCO"] * 0.85 + np.random.normal(0, 2),
        }
    elif pattern == 2:  # restrictive
        params_fn = lambda v: {
            "FVC": v["FVC"] * 0.6 + np.random.normal(0, 0.1),
            "FEV1": v["FEV1"] * 0.6 + np.random.normal(0, 0.1),
            "FEV1_FVC": v["FEV1_FVC"] * 1.0 + np.random.normal(0, 0.02),
            "FEF25_75": v["FEF25_75"] * 0.7 + np.random.normal(0, 0.2),
            "PEF": v["PEF"] * 0.7 + np.random.normal(0, 0.3),
            "FIVC": v["FIVC"] * 0.6 + np.random.normal(0, 0.1),
            "FIF50": v["FIF50"] * 0.7 + np.random.normal(0, 0.2),
            "MVV": v["MVV"] * 0.7 + np.random.normal(0, 5),
            "TLC": v["TLC"] * 0.7 + np.random.normal(0, 0.2),
            "RV": v["RV"] * 0.7 + np.random.normal(0, 0.1),
            "DLCO": v["DLCO"] * 0.7 + np.random.normal(0, 2),
        }
    else:  # mixed
        params_fn = lambda v: {
            "FVC": v["FVC"] * 0.7 + np.random.normal(0, 0.1),
            "FEV1": v["FEV1"] * 0.5 + np.random.normal(0, 0.1),
            "FEV1_FVC": v["FEV1_FVC"] * 0.75 + np.random.normal(0, 0.02),
            "FEF25_75": v["FEF25_75"] * 0.5 + np.random.normal(0, 0.2),
            "PEF": v["PEF"] * 0.6 + np.random.normal(0, 0.3),
            "FIVC": v["FIVC"] * 0.7 + np.random.normal(0, 0.1),
            "FIF50": v["FIF50"] * 0.75 + np.random.normal(0, 0.2),
            "MVV": v["MVV"] * 0.6 + np.random.normal(0, 5),
            "TLC": v["TLC"] * 0.9 + np.random.normal(0, 0.2),
            "RV": v["RV"] * 1.2 + np.random.normal(0, 0.2),
            "DLCO": v["DLCO"] * 0.75 + np.random.normal(0, 2),
        }

    for i in range(n_visits):
        visit_date = base_date + timedelta(days=np.random.randint(i*90, i*90+60))
        params = params_fn(base)
        # Add slight decline over time for realism
        decline = 1.0 - 0.01 * i
        params = {k: v * decline for k, v in params.items()}
        visits.append({
            "date": visit_date.strftime("%Y-%m-%d"),
            "params": params,
            "pattern": pattern,
            "age": 50 + i,
            "sex": "M"
        })

    return {"visits": visits}


def create_synthetic_dataset(out_dir: Path, n_per_pattern: int = 100, n_visits: int = 5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patients = {}
    for pattern_idx, pattern_name in enumerate(PATTERN_CLASSES):
        for i in range(n_per_pattern):
            pid = f"synth_{pattern_name}_{i:04d}"
            patients[pid] = generate_patient(pid, pattern_idx, n_visits)

    with open(out_dir / "patients.json", 'w') as f:
        json.dump(patients, f, indent=2)

    total = len(patients)
    print(f"Created {total} synthetic patients in {out_dir}")
    for p in PATTERN_CLASSES:
        print(f"  {p}: {n_per_pattern}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="data/spirometry_synth")
    parser.add_argument("--n-per-pattern", type=int, default=100)
    parser.add_argument("--n-visits", type=int, default=5)
    args = parser.parse_args()

    create_synthetic_dataset(Path(args.out_dir), args.n_per_pattern, args.n_visits)