#!/usr/bin/env python3
"""
Realistic chest X-ray synthesis using anatomical modeling.
Creates anatomically plausible CXR images with realistic lung fields, heart,
mediastinum, ribs, clavicles, and diaphragm. Includes pathology simulation:
consolidation, effusion, pneumothorax, nodules, cardiomegaly, etc.
"""

import json
import numpy as np
import cv2
from pathlib import Path
import argparse


IMG_SIZE = 512


def gaussian_2d(shape, center, sigma_x, sigma_y, amplitude=1.0, rotation=0):
    """2D Gaussian with rotation."""
    y, x = np.indices(shape)
    x = x - center[1]
    y = y - center[0]
    
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    x_rot = cos_r * x + sin_r * y
    y_rot = -sin_r * x + cos_r * y
    
    g = amplitude * np.exp(-(x_rot**2 / (2*sigma_x**2) + y_rot**2 / (2*sigma_y**2)))
    return g


def create_lung_fields(img_size: int) -> tuple:
    """Create left and right lung field masks with realistic anatomy."""
    mask = np.zeros((img_size, img_size), dtype=np.float32)
    
    # Left lung (smaller due to heart)
    left_center = (img_size * 0.35, img_size * 0.55)
    left_sigma = (img_size * 0.18, img_size * 0.28)
    left_lung = gaussian_2d((img_size, img_size), left_center, *left_sigma, amplitude=1.0, rotation=-0.15)
    
    # Right lung (larger, three lobes suggested)
    right_center = (img_size * 0.65, img_size * 0.55)
    right_sigma = (img_size * 0.20, img_size * 0.30)
    right_lung = gaussian_2d((img_size, img_size), right_center, *right_sigma, amplitude=1.0, rotation=0.15)
    
    # Combine
    lungs = np.maximum(left_lung, right_lung)
    
    # Add fissures (horizontal and oblique)
    # Right oblique fissure
    y, x = np.indices((img_size, img_size))
    fissure_line = (x - img_size*0.55) * 0.5 - (y - img_size*0.35)
    oblique = np.exp(-fissure_line**2 / (2 * (img_size*0.01)**2)) * 0.3
    lungs = np.minimum(lungs, 1 - oblique * 0.5)
    
    # Left oblique fissure
    fissure_line2 = (x - img_size*0.45) * -0.5 - (y - img_size*0.35)
    oblique2 = np.exp(-fissure_line2**2 / (2 * (img_size*0.01)**2)) * 0.3
    lungs = np.minimum(lungs, 1 - oblique2 * 0.5)
    
    # Horizontal fissure (right lung)
    horiz = np.exp(-(y - img_size*0.42)**2 / (2 * (img_size*0.008)**2)) * 0.2
    lungs = np.minimum(lungs, 1 - horiz * 0.5)
    
    return lungs, left_lung, right_lung


def create_heart(img_size: int, cardiomegaly: float = 0.0) -> np.ndarray:
    """Create heart shadow with optional cardiomegaly."""
    # Heart is roughly trapezoidal, right border at midline, left border at left lung
    center = (img_size * 0.55, img_size * 0.62)
    width = img_size * (0.22 + cardiomegaly * 0.15)
    height = img_size * (0.28 + cardiomegaly * 0.15)
    
    heart = gaussian_2d((img_size, img_size), center, width, height, amplitude=1.0, rotation=0.2)
    return heart


def create_diaphragm(img_size: int) -> np.ndarray:
    """Create diaphragm dome shadows."""
    y, x = np.indices((img_size, img_size))
    # Right hemidiaphragm higher (liver)
    right_dome = np.exp(-((x - img_size*0.7)**2 + (y - img_size*0.82)**2) / (2 * (img_size*0.12)**2))
    # Left hemidiaphragm lower (stomach)
    left_dome = np.exp(-((x - img_size*0.3)**2 + (y - img_size*0.86)**2) / (2 * (img_size*0.10)**2))
    return np.maximum(right_dome, left_dome) * 0.6


def create_ribs(img_size: int) -> np.ndarray:
    """Create rib shadows - curved horizontal lines."""
    ribs = np.zeros((img_size, img_size), dtype=np.float32)
    y, x = np.indices((img_size, img_size))
    
    # 6-8 visible posterior ribs per side
    for i in range(6):
        # Rib curve: higher laterally, lower medially
        rib_y = img_size * (0.30 + i * 0.07)
        # Curvature
        curve = 0.03 * img_size * np.sin(np.pi * (x / img_size)) * (1 - np.abs(x - img_size*0.5) / (img_size*0.5))
        rib_center = rib_y + curve
        
        # Rib thickness varies
        thickness = img_size * 0.006
        rib = np.exp(-(y - rib_center)**2 / (2 * thickness**2))
        ribs += rib * 0.4
    
    return np.clip(ribs, 0, 0.5)


def create_clavicles(img_size: int) -> np.ndarray:
    """Create clavicle shadows."""
    clav = np.zeros((img_size, img_size), dtype=np.float32)
    y, x = np.indices((img_size, img_size))
    
    # Left clavicle
    left_path = (y - img_size*0.22)**2 + ((x - img_size*0.25) * 1.5)**2
    left_clav = np.exp(-left_path / (2 * (img_size*0.06)**2)) * 0.7
    
    # Right clavicle
    right_path = (y - img_size*0.22)**2 + ((x - img_size*0.75) * 1.5)**2
    right_clav = np.exp(-right_path / (2 * (img_size*0.06)**2)) * 0.7
    
    return np.maximum(left_clav, right_clav)


def create_mediastinum(img_size: int) -> np.ndarray:
    """Create mediastinal structures (trachea, great vessels, etc.)."""
    med = np.zeros((img_size, img_size), dtype=np.float32)
    y, x = np.indices((img_size, img_size))
    
    # Trachea - thin vertical line
    trachea = np.exp(-(x - img_size*0.5)**2 / (2 * (img_size*0.008)**2)) * 0.4
    trachea = trachea * (y < img_size*0.45) * (y > img_size*0.15)
    
    # Aortic knob
    aortic = np.exp(-((x - img_size*0.4)**2 + (y - img_size*0.3)**2) / (2 * (img_size*0.025)**2)) * 0.5
    
    # Pulmonary trunk
    pulm = np.exp(-((x - img_size*0.52)**2 + (y - img_size*0.35)**2) / (2 * (img_size*0.02)**2)) * 0.3
    
    return np.maximum(np.maximum(trachea, aortic), pulm)


def create_soft_tissue(img_size: int) -> np.ndarray:
    """Chest wall soft tissue attenuation gradient."""
    y, x = np.indices((img_size, img_size))
    # Thicker at base, thinner at apex
    gradient = 0.1 + 0.3 * (y / img_size)**1.5
    # Lateral falloff
    lateral = np.exp(-((x - img_size*0.5) / (img_size*0.4))**2)
    return gradient * lateral * 0.3


def create_base_cxr(img_size: int = IMG_SIZE, seed: int = None) -> tuple:
    """Create a base normal CXR with all anatomical structures."""
    if seed is not None:
        np.random.seed(seed)
    
    # Start with air (black = 0 in X-ray = high transmission)
    # In X-ray: bone=white (high attenuation), air=black (low), soft tissue=gray
    img = np.zeros((img_size, img_size), dtype=np.float32)
    
    # Add anatomical structures (attenuation = brightness in X-ray)
    lungs, left_lung, right_lung = create_lung_fields(img_size)
    # Lungs are LOW attenuation (dark) - subtract from background
    # We'll build from soft tissue base and carve out lungs
    
    # Base soft tissue attenuation
    base = 0.4 + create_soft_tissue(img_size) * 0.6
    
    # Carve out lung fields (low attenuation)
    lung_mask = np.clip(lungs * 1.5, 0, 1)
    img = base * (1 - lung_mask * 0.85)  # Lungs transmit more
    
    # Add heart (high attenuation)
    heart = create_heart(img_size)
    img = np.maximum(img, heart * 0.85)
    
    # Add diaphragm
    diaphragm = create_diaphragm(img_size)
    img = np.maximum(img, diaphragm * 0.7)
    
    # Add ribs
    ribs = create_ribs(img_size)
    img += ribs * 0.3
    
    # Add clavicles
    clavicles = create_clavicles(img_size)
    img += clavicles * 0.4
    
    # Add mediastinum
    mediastinum = create_mediastinum(img_size)
    img += mediastinum * 0.4
    
    # Add noise / quantum mottle
    noise = np.random.randn(img_size, img_size) * 0.02
    img += noise
    
    # Add subtle texture to lungs (vascular markings)
    lung_texture = np.random.randn(img_size, img_size) * 0.03
    # Filter to simulate vessels
    from scipy.ndimage import gaussian_filter
    lung_texture = gaussian_filter(lung_texture, sigma=1.5)
    img += lung_texture * lung_mask * 0.5
    
    return np.clip(img, 0, 1), lung_mask, left_lung, right_lung, heart


def add_consolidation(img: np.ndarray, lung_mask: np.ndarray, 
                      side: str = 'right', severity: float = 1.0) -> np.ndarray:
    """Add consolidation (pneumonia) - increased opacity in lung region."""
    h, w = img.shape
    if side == 'right':
        center = (w * 0.65, h * 0.55)
        sigma = (w * 0.12, h * 0.10)
    else:
        center = (w * 0.35, h * 0.55)
        sigma = (w * 0.10, h * 0.09)
    
    consol = gaussian_2d((h, w), center, *sigma, amplitude=severity * 0.6)
    # Confine to lung
    consol *= lung_mask
    return np.maximum(img, consol)


def add_effusion(img: np.ndarray, side: str = 'right', severity: float = 1.0) -> np.ndarray:
    """Add pleural effusion - fluid at lung base, meniscus sign."""
    h, w = img.shape
    if side == 'right':
        # Right base - blunted costophrenic angle
        y, x = np.indices((h, w))
        # Meniscus curve
        angle_center = (w * 0.85, h * 0.85)
        dist = np.sqrt((x - angle_center[0])**2 + (y - angle_center[1])**2)
        # Fluid fills from bottom up
        fluid_level = h * (0.75 - severity * 0.15)
        effusion = np.where(y > fluid_level, 
                           np.exp(-(dist / (w*0.15))**2) * severity * 0.7, 0)
        # Meniscus: curved upper border
        meniscus = np.exp(-(y - fluid_level)**2 / (2 * (h*0.02)**2)) * 0.5
        effusion += meniscus * (dist < w*0.2)
    else:
        y, x = np.indices((h, w))
        angle_center = (w * 0.15, h * 0.85)
        dist = np.sqrt((x - angle_center[0])**2 + (y - angle_center[1])**2)
        fluid_level = h * (0.75 - severity * 0.15)
        effusion = np.where(y > fluid_level,
                           np.exp(-(dist / (w*0.15))**2) * severity * 0.7, 0)
        meniscus = np.exp(-(y - fluid_level)**2 / (2 * (h*0.02)**2)) * 0.5
        effusion += meniscus * (dist < w*0.2)
    
    return np.maximum(img, effusion)


def add_pneumothorax(img: np.ndarray, lung_mask: np.ndarray, 
                     side: str = 'right', severity: float = 1.0) -> np.ndarray:
    """Add pneumothorax - visible pleural line with absent lung markings."""
    h, w = img.shape
    if side == 'right':
        # Visible pleural line parallel to chest wall
        y, x = np.indices((h, w))
        pleural_line = np.exp(-(x - w*0.82)**2 / (2 * (w*0.008)**2)) * severity * 0.8
        # Lung collapsed medially
        collapse_mask = (x < w*0.82) & lung_mask.astype(bool)
        img = np.where(collapse_mask, img * 0.3, img)  # Reduce markings
        img += pleural_line
    else:
        y, x = np.indices((h, w))
        pleural_line = np.exp(-(x - w*0.18)**2 / (2 * (w*0.008)**2)) * severity * 0.8
        collapse_mask = (x > w*0.18) & lung_mask.astype(bool)
        img = np.where(collapse_mask, img * 0.3, img)
        img += pleural_line
    
    return np.clip(img, 0, 1)


def add_nodule(img: np.ndarray, lung_mask: np.ndarray, 
               position: tuple = None, size_mm: float = 10) -> np.ndarray:
    """Add solitary pulmonary nodule."""
    h, w = img.shape
    if position is None:
        # Random position within lung
        lung_coords = np.argwhere(lung_mask > 0.3)
        if len(lung_coords) == 0:
            return img
        y, x = lung_coords[np.random.randint(len(lung_coords))]
    else:
        y, x = position
    
    sigma_px = (size_mm / 25.4) * (w / 35) * 25.4 / 10  # Rough conversion
    nodule = gaussian_2d((h, w), (x, y), sigma_px, sigma_px, amplitude=0.7)
    nodule *= lung_mask  # Only in lung
    return np.maximum(img, nodule)


def add_cardiomegaly(img: np.ndarray, severity: float = 1.0) -> np.ndarray:
    """Enlarge heart shadow."""
    h, w = img.shape
    heart = create_heart(w, cardiomegaly=severity)
    return np.maximum(img, heart * 0.85)


def add_fibrosis(img: np.ndarray, lung_mask: np.ndarray, severity: float = 1.0) -> np.ndarray:
    """Interstitial fibrosis - reticular pattern, honeycombing."""
    h, w = img.shape
    # Reticular pattern: fine linear opacities
    texture = np.random.randn(h, w) * 0.05
    from scipy.ndimage import gaussian_filter
    # Anisotropic filtering for linear pattern
    texture_x = gaussian_filter(texture, sigma=(0.5, 3.0))
    texture_y = gaussian_filter(texture, sigma=(3.0, 0.5))
    reticular = np.maximum(texture_x, texture_y) * severity * 0.4
    reticular *= lung_mask
    return np.maximum(img, reticular)


def add_emphysema(img: np.ndarray, lung_mask: np.ndarray, severity: float = 1.0) -> np.ndarray:
    """Emphysema - hyperlucent lungs, flattened diaphragm, bullae."""
    h, w = img.shape
    # Increase lung lucency (decrease attenuation)
    img = np.where(lung_mask > 0.3, img * (1 - severity * 0.3), img)
    # Flatten diaphragm
    y, x = np.indices((h, w))
    flat_diaphragm = np.exp(-(y - h*0.88)**2 / (2 * (h*0.02)**2)) * severity * 0.3
    img += flat_diaphragm
    # Bullae - large lucent spaces
    for _ in range(int(severity * 3)):
        by = np.random.randint(int(h*0.2), int(h*0.7))
        bx = np.random.randint(int(w*0.2), int(w*0.8))
        if lung_mask[by, bx] > 0.3:
            bulla = gaussian_2d((h, w), (bx, by), w*0.03, h*0.03, amplitude=-0.5)
            bulla *= lung_mask
            img = np.maximum(img, img + bulla)  # Negative = more lucent
    return np.clip(img, 0, 1)


def generate_case(out_path: Path, case_type: str, seed: int = None) -> dict:
    """Generate a realistic CXR case."""
    if seed is not None:
        np.random.seed(seed)
    
    img, lung_mask, left_lung, right_lung, heart = create_base_cxr(IMG_SIZE, seed)
    
    label_info = {"pathology": case_type}
    
    if case_type == 'normal':
        pass
    elif case_type == 'consolidation_right':
        img = add_consolidation(img, lung_mask, 'right', np.random.uniform(0.7, 1.0))
        label_info["location"] = "right"
        label_info["severity"] = "moderate"
    elif case_type == 'consolidation_left':
        img = add_consolidation(img, lung_mask, 'left', np.random.uniform(0.7, 1.0))
        label_info["location"] = "left"
        label_info["severity"] = "moderate"
    elif case_type == 'consolidation_bilateral':
        img = add_consolidation(img, lung_mask, 'right', np.random.uniform(0.5, 0.8))
        img = add_consolidation(img, lung_mask, 'left', np.random.uniform(0.5, 0.8))
        label_info["location"] = "bilateral"
    elif case_type == 'effusion_right':
        img = add_effusion(img, 'right', np.random.uniform(0.6, 1.0))
        label_info["location"] = "right"
    elif case_type == 'effusion_left':
        img = add_effusion(img, 'left', np.random.uniform(0.6, 1.0))
        label_info["location"] = "left"
    elif case_type == 'effusion_bilateral':
        img = add_effusion(img, 'right', np.random.uniform(0.4, 0.7))
        img = add_effusion(img, 'left', np.random.uniform(0.4, 0.7))
        label_info["location"] = "bilateral"
    elif case_type == 'pneumothorax_right':
        img = add_pneumothorax(img, lung_mask, 'right', np.random.uniform(0.7, 1.0))
        label_info["location"] = "right"
    elif case_type == 'pneumothorax_left':
        img = add_pneumothorax(img, lung_mask, 'left', np.random.uniform(0.7, 1.0))
        label_info["location"] = "left"
    elif case_type == 'nodule_solitary':
        img = add_nodule(img, lung_mask, size_mm=np.random.uniform(8, 20))
        label_info["size_mm"] = "8-20"
    elif case_type == 'nodules_multiple':
        for _ in range(np.random.randint(3, 8)):
            img = add_nodule(img, lung_mask, size_mm=np.random.uniform(5, 15))
        label_info["count"] = "multiple"
    elif case_type == 'cardiomegaly':
        img = add_cardiomegaly(img, np.random.uniform(0.5, 1.0))
        label_info["severity"] = "moderate-severe"
    elif case_type == 'fibrosis':
        img = add_fibrosis(img, lung_mask, np.random.uniform(0.5, 1.0))
        label_info["pattern"] = "reticular"
    elif case_type == 'emphysema':
        img = add_emphysema(img, lung_mask, np.random.uniform(0.5, 1.0))
        label_info["severity"] = "moderate"
    elif case_type == 'copd':  # Emphysema + chronic bronchitis signs
        img = add_emphysema(img, lung_mask, np.random.uniform(0.4, 0.7))
        # Prominent bronchial walls
        bronchial = create_mediastinum(IMG_SIZE) * 1.5
        img += bronchial * lung_mask * 0.3
        label_info["components"] = "emphysema + bronchitis"
    elif case_type == 'interstitial_edema':  # Kerley B lines, perihilar haze
        h, w = img.shape
        # Perihilar haze
        haze = gaussian_2d((h, w), (w*0.5, h*0.4), w*0.12, h*0.1, amplitude=0.4)
        img += haze * lung_mask * 0.6
        # Kerley B lines - short horizontal lines at bases
        y, x = np.indices((h, w))
        for i in range(5):
            line_y = h * (0.75 + i * 0.02)
            kerley = np.exp(-(y - line_y)**2 / (2 * (h*0.003)**2)) * 0.3
            kerley *= (x > w*0.2) & (x < w*0.8)
            img += kerley * lung_mask * 0.7
        img = np.clip(img, 0, 1)
        label_info["pattern"] = "alveolar/interstitial"
    else:
        raise ValueError(f"Unknown case type: {case_type}")
    
    # Final processing: simulate X-ray physics
    # Add quantum noise (Poisson)
    photons = 10000
    img = np.random.poisson(img * photons) / photons
    
    # Add subtle motion blur
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, sigma=0.4)
    
    # Normalize and convert
    img = np.clip(img, 0, 1)
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Apply histogram equalization for realistic X-ray look
    img_uint8 = cv2.equalizeHist(img_uint8)
    
    cv2.imwrite(str(out_path), img_uint8)
    
    return {"path": str(out_path), "pathology": case_type, **label_info}


CASE_TYPES = [
    'normal',
    'consolidation_right', 'consolidation_left', 'consolidation_bilateral',
    'effusion_right', 'effusion_left', 'effusion_bilateral',
    'pneumothorax_right', 'pneumothorax_left',
    'nodule_solitary', 'nodules_multiple',
    'cardiomegaly',
    'fibrosis',
    'emphysema',
    'copd',
    'interstitial_edema',
]

# Clinical label mapping for the classifier
CLINICAL_LABELS = {
    'normal': ['No Finding'],
    'consolidation_right': ['Consolidation', 'Infiltration'],
    'consolidation_left': ['Consolidation', 'Infiltration'],
    'consolidation_bilateral': ['Consolidation', 'Infiltration'],
    'effusion_right': ['Effusion'],
    'effusion_left': ['Effusion'],
    'effusion_bilateral': ['Effusion'],
    'pneumothorax_right': ['Pneumothorax'],
    'pneumothorax_left': ['Pneumothorax'],
    'nodule_solitary': ['Nodule', 'Mass'],
    'nodules_multiple': ['Nodule', 'Mass'],
    'cardiomegaly': ['Cardiomegaly'],
    'fibrosis': ['Fibrosis', 'Pleural_Thickening'],
    'emphysema': ['Emphysema'],
    'copd': ['Emphysema', 'Cardiomegaly'],
    'interstitial_edema': ['Edema', 'Infiltration'],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="test-data/cxr_realistic")
    parser.add_argument("--n-per-type", type=int, default=2)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    labels = {}
    for case_type in CASE_TYPES:
        for i in range(args.n_per_type):
            fname = f"{case_type}_{i+1:02d}.png"
            fpath = out_dir / fname
            meta = generate_case(fpath, case_type, seed=hash(fname) % 2**32)
            labels[fname] = {
                "clinical_labels": CLINICAL_LABELS.get(case_type, []),
                "synthesis_type": case_type,
                **{k: v for k, v in meta.items() if k != 'path'}
            }
    
    with open(out_dir / "labels.json", 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"Generated {len(labels)} realistic CXR images in {out_dir}")
    for ct in CASE_TYPES:
        print(f"  {ct}: {args.n_per_type} samples")