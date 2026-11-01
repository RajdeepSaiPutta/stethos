#!/usr/bin/env python3
"""
CXR Model Training Script
Trains DenseNet121 classifier (14 CheXNet pathologies) + MobileNetV3 U-Net segmenter (5 classes)
Saves checkpoints to python/models/ for the stethos sidecar.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
from torchvision.models import densenet121, DenseNet121_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------------------------------------------------
# Model definitions (mirror plugins/cxr.py exactly so weights load)
# ---------------------------------------------------------------------

CHEXNET_PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]

LUNG_CLASSES = ["background", "left_lung", "right_lung", "heart", "trachea"]
NUM_CLASSES = len(CHEXNET_PATHOLOGIES)
NUM_SEG_CLASSES = len(LUNG_CLASSES)


class CXRClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)
        self.features = backbone.features
        num_features = backbone.classifier.in_features  # 1024
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class MedSAMSegmenter(nn.Module):
    def __init__(self, num_classes: int = NUM_SEG_CLASSES, pretrained_encoder: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained_encoder else None
        backbone = mobilenet_v3_small(weights=weights)
        self.encoder = backbone.features

        self.decoder = nn.Sequential(
            nn.Conv2d(576, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.decoder(features)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class CXRDataset(Dataset):
    """
    Expects a directory structure:
    data_dir/
      images/           # PNG/JPG/DICOM files
      labels.json       # {"image_filename": {"pathologies": [0/1]*14, "seg_mask": "path/to/mask.png"}}
      segmentation/     # optional: mask PNGs (5-class indexed)
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 512,
        augments: bool = True,
        seg_available: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.seg_available = seg_available

        labels_path = self.data_dir / "labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.json not found in {data_dir}")
        with open(labels_path) as f:
            self.labels = json.load(f)

        self.image_files = list(self.labels.keys())
        print(f"[{split}] Loaded {len(self.image_files)} images")

        # Albumentations transforms
        if augments and split == "train":
            self.transform = A.Compose([
                A.RandomRotate90(p=0.3),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussNoise(std_range=(0.1, 0.2), p=0.2),
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]),
                ToTensorV2(),
            ])

        # Segmentation transform (no color augs, nearest neighbor for mask)
        if seg_available:
            self.seg_transform = A.Compose([
                A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
            ], additional_targets={'mask': 'mask'})

    def _load_image(self, fname: str) -> np.ndarray:
        path = self.data_dir / "images" / fname
        if path.suffix.lower() in ['.dcm', '.dicom']:
            import pydicom
            ds = pydicom.dcmread(str(path))
            img = ds.pixel_array.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)
        else:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        # Replicate to 3 channels for ImageNet backbones
        img = np.repeat(img[..., np.newaxis], 3, axis=2)
        return img

    def _load_mask(self, fname: str) -> Optional[np.ndarray]:
        if not self.seg_available:
            return None
        mask_path = self.data_dir / "segmentation" / fname
        if not mask_path.exists():
            mask_path = self.data_dir / "segmentation" / (Path(fname).stem + ".png")
        if not mask_path.exists():
            return None
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return mask  # Already class indices 0-4

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict:
        fname = self.image_files[idx]
        label = self.labels[fname]

        img = self._load_image(fname)
        transformed = self.transform(image=img)
        img_tensor = transformed['image']  # [3, H, W]

        # Classification target
        path_target = torch.tensor(label["pathologies"], dtype=torch.float32)  # [14]

        out = {"image": img_tensor, "pathologies": path_target, "fname": fname}

        if self.seg_available:
            mask = self._load_mask(fname)
            if mask is not None:
                transformed_seg = self.seg_transform(image=img, mask=mask)
                # mask stays as class indices [H, W]
                mask_tensor = torch.from_numpy(transformed_seg['mask']).long()
                out["seg_mask"] = mask_tensor

        return out


# ---------------------------------------------------------------------
# Losses & Metrics
# ---------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for multi-label classification to handle class imbalance."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: [B, C, H, W] logits, targets: [B, H, W] class indices
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def compute_auroc_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute AUROC for each of the 14 classes."""
    from sklearn.metrics import roc_auc_score
    aurocs = []
    for i in range(y_true.shape[1]):
        try:
            aurocs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        except ValueError:
            aurocs.append(float('nan'))
    return np.array(aurocs)


def compute_dice_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> np.ndarray:
    """Compute Dice coefficient per segmentation class."""
    pred = pred.argmax(dim=1)  # [B, H, W]
    dices = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        inter = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        dice = (2 * inter + 1e-6) / (union + 1e-6)
        dices.append(dice.mean().item())
    return np.array(dices)


# ---------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------

def train_epoch(
    classifier: nn.Module,
    segmenter: nn.Module,
    loader: DataLoader,
    cls_opt: torch.optim.Optimizer,
    seg_opt: torch.optim.Optimizer,
    cls_scaler: torch.amp.GradScaler,
    seg_scaler: torch.amp.GradScaler,
    device: torch.device,
    cls_criterion: nn.Module,
    seg_criterion_ce: nn.Module,
    seg_criterion_dice: nn.Module,
    lambda_seg: float = 1.0,
    lambda_dice: float = 1.0,
    accum_steps: int = 1,
) -> Dict[str, float]:
    classifier.train()
    segmenter.train()

    total_cls_loss = 0.0
    total_seg_loss = 0.0
    total_dice_loss = 0.0
    n_batches = 0

    for i, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        img = batch['image'].to(device, non_blocking=True)
        path_target = batch['pathologies'].to(device, non_blocking=True)

        # ---- Classifier ----
        with torch.amp.autocast('cuda'):
            cls_logits = classifier(img)
            cls_loss = cls_criterion(cls_logits, path_target)
            cls_loss = cls_loss / accum_steps

        cls_scaler.scale(cls_loss).backward()

        # ---- Segmenter (if masks available) ----
        seg_loss = torch.tensor(0.0, device=device)
        dice_loss = torch.tensor(0.0, device=device)
        if 'seg_mask' in batch:
            seg_mask = batch['seg_mask'].to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                seg_logits = segmenter(img)
                ce = seg_criterion_ce(seg_logits, seg_mask)
                dice = seg_criterion_dice(seg_logits, seg_mask)
                seg_loss = (lambda_seg * ce + lambda_dice * dice) / accum_steps

            seg_scaler.scale(seg_loss).backward()

        if (i + 1) % accum_steps == 0:
            cls_scaler.step(cls_opt)
            cls_scaler.update()
            cls_opt.zero_grad(set_to_none=True)

            if 'seg_mask' in batch:
                seg_scaler.step(seg_opt)
                seg_scaler.update()
                seg_opt.zero_grad(set_to_none=True)

        total_cls_loss += cls_loss.item() * accum_steps
        if 'seg_mask' in batch:
            total_seg_loss += seg_loss.item() * accum_steps
        n_batches += 1

    return {
        "cls_loss": total_cls_loss / n_batches,
        "seg_loss": total_seg_loss / max(1, n_batches),
    }


@torch.no_grad()
def validate(
    classifier: nn.Module,
    segmenter: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cls_criterion: nn.Module,
    seg_criterion_ce: nn.Module,
    seg_criterion_dice: nn.Module,
    lambda_seg: float = 1.0,
    lambda_dice: float = 1.0,
) -> Dict[str, float]:
    classifier.eval()
    segmenter.eval()

    all_path_true = []
    all_path_pred = []
    total_cls_loss = 0.0
    total_seg_loss = 0.0
    total_dice_loss = 0.0
    all_seg_dices = []
    n_batches = 0

    for batch in tqdm(loader, desc="Validating", leave=False):
        img = batch['image'].to(device, non_blocking=True)
        path_target = batch['pathologies'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            cls_logits = classifier(img)
            cls_loss = cls_criterion(cls_logits, path_target)

            cls_probs = torch.sigmoid(cls_logits)
            all_path_true.append(path_target.cpu().numpy())
            all_path_pred.append(cls_probs.cpu().numpy())

        total_cls_loss += cls_loss.item()

        if 'seg_mask' in batch:
            seg_mask = batch['seg_mask'].to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                seg_logits = segmenter(img)
                ce = seg_criterion_ce(seg_logits, seg_mask)
                dice = seg_criterion_dice(seg_logits, seg_mask)
                seg_loss = lambda_seg * ce + lambda_dice * dice

            total_seg_loss += seg_loss.item()

            dices = compute_dice_per_class(seg_logits, seg_mask, NUM_SEG_CLASSES)
            all_seg_dices.append(dices)

        n_batches += 1

    all_path_true = np.vstack(all_path_true)
    all_path_pred = np.vstack(all_path_pred)

    aurocs = compute_auroc_per_class(all_path_true, all_path_pred)
    mean_auroc = np.nanmean(aurocs)

    results = {
        "cls_loss": total_cls_loss / n_batches,
        "mean_auroc": mean_auroc,
        "aurocs": dict(zip(CHEXNET_PATHOLOGIES, aurocs.tolist())),
    }

    if all_seg_dices:
        mean_dices = np.mean(all_seg_dices, axis=0)
        results["seg_loss"] = total_seg_loss / n_batches
        results["dice_per_class"] = dict(zip(LUNG_CLASSES, mean_dices.tolist()))
        results["mean_dice"] = float(np.mean(mean_dices[1:]))  # exclude background

    return results


def save_checkpoint(
    classifier: nn.Module,
    segmenter: nn.Module,
    cls_opt: torch.optim.Optimizer,
    seg_opt: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    out_dir: Path,
    name: str
):
    out_dir.mkdir(parents=True, exist_ok=True)

    if "classifier" in name:
        path = out_dir / "cxr_classifier.pt"
        torch.save({"model_state_dict": classifier.state_dict()}, path)
        print(f"Saved classifier: {path}")
    elif "segmenter" in name:
        path = out_dir / "cxr_segmenter.pt"
        torch.save({"model_state_dict": segmenter.state_dict()}, path)
        print(f"Saved segmenter: {path}")
    else:
        # latest - save both
        torch.save({"model_state_dict": classifier.state_dict()}, out_dir / "cxr_classifier.pt")
        torch.save({"model_state_dict": segmenter.state_dict()}, out_dir / "cxr_segmenter.pt")
        print(f"Saved both to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train CXR classifier + segmenter")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root (with images/, labels.json, segmentation/)")
    parser.add_argument("--output-dir", type=str, default="python/models", help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained backbones")
    parser.add_argument("--seg-weight", type=float, default=1.0, help="CE loss weight for segmentation")
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Dice loss weight for segmentation")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Dataset
    full_ds = CXRDataset(
        args.data_dir,
        split="train",
        img_size=args.img_size,
        augments=True,
        seg_available=(Path(args.data_dir) / "segmentation").exists()
    )

    val_size = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    # Disable augs for val
    val_ds.dataset.seg_available = full_ds.seg_available  # type: ignore
    # Create a fresh val dataset without augs
    val_ds_clean = CXRDataset(
        args.data_dir,
        split="val",
        img_size=args.img_size,
        augments=False,
        seg_available=full_ds.seg_available
    )
    val_ds_clean.image_files = [full_ds.image_files[i] for i in val_ds.indices]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds_clean, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )

    # Models
    classifier = CXRClassifier(num_classes=NUM_CLASSES, pretrained=args.pretrained).to(device)
    segmenter = MedSAMSegmenter(num_classes=NUM_SEG_CLASSES, pretrained_encoder=args.pretrained).to(device)

    # Optimizers
    cls_opt = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    seg_opt = torch.optim.AdamW(segmenter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Schedulers
    cls_sched = torch.optim.lr_scheduler.CosineAnnealingLR(cls_opt, T_max=args.epochs)
    seg_sched = torch.optim.lr_scheduler.CosineAnnealingLR(seg_opt, T_max=args.epochs)

    # Loss
    cls_criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)
    seg_criterion_ce = nn.CrossEntropyLoss(ignore_index=255).to(device)
    seg_criterion_dice = DiceLoss().to(device)

    # AMP scalers
    cls_scaler = torch.amp.GradScaler('cuda')
    seg_scaler = torch.amp.GradScaler('cuda')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_auroc = 0.0
    best_dice = 0.0

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds_clean)}")
    print(f"Segmentation available: {full_ds.seg_available}\n")

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_metrics = train_epoch(
            classifier, segmenter, train_loader,
            cls_opt, seg_opt, cls_scaler, seg_scaler,
            device, cls_criterion, seg_criterion_ce, seg_criterion_dice,
            lambda_seg=args.seg_weight, lambda_dice=args.dice_weight,
            accum_steps=args.accum_steps
        )
        print(f"Train: cls_loss={train_metrics['cls_loss']:.4f}", end="")
        if full_ds.seg_available:
            print(f", seg_loss={train_metrics['seg_loss']:.4f}")
        else:
            print()

        val_metrics = validate(
            classifier, segmenter, val_loader,
            device, cls_criterion, seg_criterion_ce, seg_criterion_dice,
            lambda_seg=args.seg_weight, lambda_dice=args.dice_weight
        )

        print(f"Val:   cls_loss={val_metrics['cls_loss']:.4f}, mean_AUROC={val_metrics['mean_auroc']:.4f}")
        for name, auroc in val_metrics['aurocs'].items():
            print(f"  {name}: {auroc:.4f}")

        if full_ds.seg_available:
            print(f"       seg_loss={val_metrics['seg_loss']:.4f}, mean_Dice={val_metrics['mean_dice']:.4f}")
            for name, dice in val_metrics['dice_per_class'].items():
                print(f"  {name}: {dice:.4f}")

        cls_sched.step()
        seg_sched.step()

        # Save best
        if val_metrics['mean_auroc'] > best_auroc:
            best_auroc = val_metrics['mean_auroc']
            save_checkpoint(classifier, segmenter, cls_opt, seg_opt, epoch, val_metrics, out_dir, "cxr_classifier_best")

        if full_ds.seg_available and val_metrics['mean_dice'] > best_dice:
            best_dice = val_metrics['mean_dice']
            save_checkpoint(classifier, segmenter, cls_opt, seg_opt, epoch, val_metrics, out_dir, "cxr_segmenter_best")

        # Always save latest
        save_checkpoint(classifier, segmenter, cls_opt, seg_opt, epoch, val_metrics, out_dir, "cxr_latest")

    print(f"\nTraining complete. Best AUROC: {best_auroc:.4f}", end="")
    if full_ds.seg_available:
        print(f", Best Dice: {best_dice:.4f}")
    else:
        print()
    print(f"Checkpoints in: {out_dir}")


if __name__ == "__main__":
    main()