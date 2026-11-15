#!/usr/bin/env python3
"""
Realistic spirometry synthesis based on physiological flow-volume curves.
Generates clinically accurate FVC maneuvers with proper flow-volume loop morphology
for normal, obstructive, restrictive, and mixed patterns. Includes within-maneuver
variability and common artifacts.
"""

import json
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from typing import Dict, List


# Standard PFT parameters
PFT_PARAMETERS = [
    "FVC", "FEV1", "FEV1_FVC", "FEF25_75", "FEF50", "FEF75",
    "PEF", "FIVC", "FIF50", "PIF", "MVV", "TLC", "RV", "DLCO"
]

PATTERN_CLASSES = ["normal", "obstructive", "restrictive", "mixed"]


def flow_volume_curve(pattern: str, seed: int = None, n_points: int = 200) -> Dict:
    """Generate a realistic flow-volume loop for a given pattern.
    Returns: volume (L), flow (L/s) arrays for both inspiration and expiration.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Base predicted values for adult male, 175cm, 70kg, 45yo
    pred = {
        "FVC": 4.8, "FEV1": 3.9, "PEF": 9.5, "FEF25_75": 4.2,
        "FIVC": 4.8, "FIF50": 5.5, "PIF": 7.0,
        "TLC": 6.2, "RV": 1.5, "DLCO": 28
    }
    
    if pattern == "normal":
        fvc = pred["FVC"] * np.random.normal(1.0, 0.08)
        fev1 = pred["FEV1"] * np.random.normal(1.0, 0.07)
        pef = pred["PEF"] * np.random.normal(1.0, 0.10)
        fef50 = pred["FEF25_75"] * np.random.normal(1.0, 0.12)
        tlc = pred["TLC"] * np.random.normal(1.0, 0.06)
        rv = pred["RV"] * np.random.normal(1.0, 0.10)
        dlco = pred["DLCO"] * np.random.normal(1.0, 0.10)
        
        # Normal flow-volume: rapid rise to PEF, linear decline, smooth inspiration
        v_exp = np.linspace(0, fvc, n_points)
        # Expiratory flow: PEF at start, then roughly linear to 0 at FVC
        # With slight convexity (normal)
        flow_exp = pef * (1 - v_exp / fvc) ** 0.7
        # Add slight curvilinearity
        flow_exp *= (1 + 0.15 * np.sin(np.pi * v_exp / fvc))
        
        # Inspiratory: symmetric but slightly lower peak
        v_insp = np.linspace(fvc, 0, n_points)
        pif = pred["PIF"] * np.random.normal(1.0, 0.10)
        flow_insp = -pif * (1 - (fvc - v_insp) / fvc) ** 0.6  # Negative = inspiration
        flow_insp *= (1 + 0.1 * np.sin(np.pi * (fvc - v_insp) / fvc))
        
    elif pattern == "obstructive":
        # Reduced FEV1/FVC, scooped expiratory limb, reduced flows at low volumes
        fvc = pred["FVC"] * np.random.normal(0.95, 0.10)
        fev1 = pred["FEV1"] * np.random.normal(0.55, 0.12)
        pef = pred["PEF"] * np.random.normal(0.75, 0.12)
        fef50 = pred["FEF25_75"] * np.random.normal(0.35, 0.15)
        tlc = pred["TLC"] * np.random.normal(1.15, 0.08)  # Hyperinflation
        rv = pred["RV"] * np.random.normal(1.6, 0.15)      # Air trapping
        dlco = pred["DLCO"] * np.random.normal(0.85, 0.12)
        
        v_exp = np.linspace(0, fvc, n_points)
        # Obstructive: lower PEF, rapid early decline, then "scooped" concave curve
        # FEV1/FVC ~ 0.55-0.65
        flow_exp = pef * (1 - v_exp / fvc) ** 1.8
        # Add concavity (scooping)
        flow_exp *= (1 - 0.5 * np.sin(np.pi * v_exp / fvc) ** 2)
        
        v_insp = np.linspace(fvc, 0, n_points)
        pif = pred["PIF"] * np.random.normal(0.8, 0.12)
        flow_insp = -pif * (1 - (fvc - v_insp) / fvc) ** 1.2
        
    elif pattern == "restrictive":
        # Reduced volumes, preserved FEV1/FVC, steep expiratory decline
        fvc = pred["FVC"] * np.random.normal(0.60, 0.08)
        fev1 = pred["FEV1"] * np.random.normal(0.60, 0.08)
        pef = pred["PEF"] * np.random.normal(0.70, 0.10)
        fef50 = pred["FEF25_75"] * np.random.normal(0.75, 0.12)
        tlc = pred["TLC"] * np.random.normal(0.65, 0.07)
        rv = pred["RV"] * np.random.normal(0.65, 0.10)
        dlco = pred["DLCO"] * np.random.normal(0.70, 0.10)
        
        v_exp = np.linspace(0, fvc, n_points)
        # Restrictive: steeper linear decline, smaller volume
        flow_exp = pef * (1 - v_exp / fvc) ** 0.5
        flow_exp *= (1 + 0.1 * np.sin(np.pi * v_exp / fvc))
        
        v_insp = np.linspace(fvc, 0, n_points)
        pif = pred["PIF"] * np.random.normal(0.7, 0.10)
        flow_insp = -pif * (1 - (fvc - v_insp) / fvc) ** 0.4
        
    elif pattern == "mixed":
        # Features of both: reduced FVC + scooped expiratory limb
        fvc = pred["FVC"] * np.random.normal(0.70, 0.08)
        fev1 = pred["FEV1"] * np.random.normal(0.45, 0.10)
        pef = pred["PEF"] * np.random.normal(0.65, 0.12)
        fef50 = pred["FEF25_75"] * np.random.normal(0.45, 0.15)
        tlc = pred["TLC"] * np.random.normal(0.90, 0.08)
        rv = pred["RV"] * np.random.normal(1.2, 0.15)
        dlco = pred["DLCO"] * np.random.normal(0.75, 0.10)
        
        v_exp = np.linspace(0, fvc, n_points)
        flow_exp = pef * (1 - v_exp / fvc) ** 1.4
        flow_exp *= (1 - 0.4 * np.sin(np.pi * v_exp / fvc) ** 2)
        
        v_insp = np.linspace(fvc, 0, n_points)
        pif = pred["PIF"] * np.random.normal(0.75, 0.10)
        flow_insp = -pif * (1 - (fvc - v_insp) / fvc) ** 0.9
        
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    # Add within-maneuver variability (noise, slight hysteresis)
    flow_exp += np.random.normal(0, pef * 0.02, n_points)
    flow_insp += np.random.normal(0, abs(pif) * 0.02, n_points)
    
    # Ensure flow is zero at volume extremes
    flow_exp[0] = pef
    flow_exp[-1] = 0
    flow_insp[0] = 0
    flow_insp[-1] = -pif
    
    # Calculate derived parameters
    fev1_fvc = fev1 / fvc
    fef25 = np.interp(fvc * 0.25, v_exp, flow_exp)
    fef75 = np.interp(fvc * 0.75, v_exp, flow_exp)
    
    return {
        "v_exp": v_exp.tolist(),
        "flow_exp": flow_exp.tolist(),
        "v_insp": v_insp.tolist(),
        "flow_insp": flow_insp.tolist(),
        "parameters": {
            "FVC": round(float(fvc), 2),
            "FEV1": round(float(fev1), 2),
            "FEV1_FVC": round(float(fev1_fvc), 3),
            "PEF": round(float(pef), 2),
            "FEF25": round(float(fef25), 2),
            "FEF50": round(float(fef50), 2),
            "FEF75": round(float(fef75), 2),
            "FEF25_75": round(float((fef25 + fef50 + fef75) / 3), 2),
            "PIF": round(float(pif), 2),
            "FIVC": round(float(fvc), 2),
            "TLC": round(float(tlc), 2),
            "RV": round(float(rv), 2),
            "DLCO": round(float(dlco), 1),
        }
    }


def add_artifacts(curve: Dict, artifact_prob: float = 0.15) -> Dict:
    """Add common spirometry artifacts."""
    v_exp = np.array(curve["v_exp"])
    flow_exp = np.array(curve["flow_exp"])
    v_insp = np.array(curve["v_insp"])
    flow_insp = np.array(curve["flow_insp"])
    
    if np.random.random() < artifact_prob:
        # Cough artifact: sharp spike in flow
        idx = np.random.randint(len(flow_exp) // 4, 3 * len(flow_exp) // 4)
        flow_exp[idx:idx+3] += np.random.normal(0, 3.0, 3)
    
    if np.random.random() < artifact_prob:
        # Hesitation: delayed start, reduced PEF
        delay = np.random.randint(3, 8)
        flow_exp[:delay] *= np.linspace(0.2, 1.0, delay)
    
    if np.random.random() < artifact_prob:
        # Early termination: volume doesn't reach plateau
        cutoff = np.random.randint(int(0.85 * len(v_exp)), int(0.95 * len(v_exp)))
        flow_exp[cutoff:] = np.linspace(flow_exp[cutoff], 0, len(v_exp) - cutoff)
        v_exp[cutoff:] = np.linspace(v_exp[cutoff], v_exp[-1], len(v_exp) - cutoff)
        # Adjust FVC
        curve["parameters"]["FVC"] = round(float(v_exp[-1]), 2)
    
    if np.random.random() < artifact_prob:
        # Leak: inspiratory volume > expiratory volume
        leak_factor = np.random.uniform(1.02, 1.08)
        flow_insp *= leak_factor
    
    curve["v_exp"] = v_exp.tolist()
    curve["flow_exp"] = flow_exp.tolist()
    curve["v_insp"] = v_insp.tolist()
    curve["flow_insp"] = flow_insp.tolist()
    return curve


def generate_patient(patient_id: str, pattern: int, n_visits: int = 5) -> Dict:
    """Generate longitudinal spirometry for a patient."""
    pattern_name = PATTERN_CLASSES[pattern]
    base_date = datetime(2020, 1, 1)
    visits = []
    
    for i in range(n_visits):
        visit_date = base_date + timedelta(days=np.random.randint(i*90, i*90+60))
        seed = hash(f"{patient_id}_{i}") % 2**32
        curve = flow_volume_curve(pattern_name, seed)
        curve = add_artifacts(curve, artifact_prob=0.2)
        
        # Add slight decline over time
        decline = 1.0 - 0.015 * i  # 1.5% per year decline
        for k in curve["parameters"]:
            if k not in ["FEV1_FVC"]:  # Ratio may not decline linearly
                curve["parameters"][k] *= decline
        
        visits.append({
            "date": visit_date.strftime("%Y-%m-%d"),
            "age": 45 + i + np.random.randint(-2, 3),
            "sex": np.random.choice(["M", "F"]),
            "height_cm": np.random.normal(170, 8),
            "weight_kg": np.random.normal(75, 12),
            "pattern": pattern_name,
            "flow_volume": {
                "volume_exp": curve["v_exp"],
                "flow_exp": curve["flow_exp"],
                "volume_insp": curve["v_insp"],
                "flow_insp": curve["flow_insp"],
            },
            "parameters": curve["parameters"]
        })
    
    return {"visits": visits}


def generate_maneuver_csv(flow_volume: Dict, patient_info: Dict) -> str:
    """Generate CSV format for a single maneuver."""
    v_exp = flow_volume["volume_exp"]
    flow_exp = flow_volume["flow_exp"]
    v_insp = flow_volume["volume_insp"]
    flow_insp = flow_volume["flow_insp"]
    
    lines = ["time_sec,volume_L,flow_Lps,phase"]
    dt = 0.05  # 20 Hz sampling
    
    t = 0
    for v, f in zip(v_exp, flow_exp):
        lines.append(f"{t:.2f},{v:.3f},{f:.3f},expiration")
        t += dt
    
    for v, f in zip(v_insp, flow_insp):
        lines.append(f"{t:.2f},{v:.3f},{f:.3f},inspiration")
        t += dt
    
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="test-data/spirometry_realistic")
    parser.add_argument("--n-per-pattern", type=int, default=8)
    parser.add_argument("--n-visits", type=int, default=4)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    json_dir = out_dir / "json"
    json_dir.mkdir(exist_ok=True)
    
    patients = {}
    csv_files = {}
    
    for pattern_idx, pattern_name in enumerate(PATTERN_CLASSES):
        for i in range(args.n_per_pattern):
            pid = f"{pattern_name}_patient_{i+1:03d}"
            patient = generate_patient(pid, pattern_idx, args.n_visits)
            patients[pid] = patient
            
            # Save latest visit as CSV and JSON
            latest = patient["visits"][-1]
            csv_content = generate_maneuver_csv(latest["flow_volume"], latest)
            csv_path = csv_dir / f"{pid}.csv"
            csv_path.write_text(csv_content)
            
            json_path = json_dir / f"{pid}.json"
            json_path.write_text(json.dumps(latest, indent=2))
            
            csv_files[pid] = str(csv_path)
    
    # Summary labels
    labels = {}
    for pid, patient in patients.items():
        latest = patient["visits"][-1]
        labels[pid] = {
            "pattern": latest["pattern"],
            "parameters": latest["parameters"],
            "csv_file": csv_files[pid],
            "age": latest["age"],
            "sex": latest["sex"],
            "height_cm": latest["height_cm"],
            "weight_kg": latest["weight_kg"],
        }
    
    with open(out_dir / "labels.json", 'w') as f:
        json.dump(labels, f, indent=2)
    
    with open(out_dir / "patients.json", 'w') as f:
        json.dump(patients, f, indent=2)
    
    print(f"Generated realistic spirometry data in {out_dir}")
    print(f"  Patients: {len(patients)} ({args.n_per_pattern} per pattern)")
    print(f"  Visits per patient: {args.n_visits}")
    print(f"  CSV files: {len(csv_files)}")
    for p in PATTERN_CLASSES:
        print(f"  {p}: {args.n_per_pattern} patients")