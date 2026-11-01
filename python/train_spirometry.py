#!/usr/bin/env python3
"""
Spirometry Model Training Script
Trains Transformer on longitudinal PFT data for pattern classification (Normal, Obstructive, Restrictive, Mixed)
and parameter prediction. Saves checkpoint to python/models/spirometry.pt
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib

# ---------------------------------------------------------------------
# Model (mirror plugins/spirometry.py exactly)
# ---------------------------------------------------------------------

PFT_PARAMETERS = [
    "FVC", "FEV1", "FEV1_FVC", "FEF25_75", "PEF",
    "FIVC", "FIF50", "MVV", "TLC", "RV", "DLCO"
]
INPUT_DIM = len(PFT_PARAMETERS)
PATTERN_CLASSES = ["normal", "obstructive", "restrictive", "mixed"]
NUM_PATTERNS = len(PATTERN_CLASSES)
MAX_SEQ_LEN = 50


class SpirometryTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        num_patterns: int = NUM_PATTERNS,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, MAX_SEQ_LEN, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, INPUT_DIM)
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_patterns)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # x: [batch, seq_len, input_dim]
        seq_len = x.shape[1]
        x = self.input_proj(x) + self.pos_encoding[:, :seq_len, :]
        if mask is not None:
            x = self.encoder(x, src_key_padding_mask=mask)
        else:
            x = self.encoder(x)
        last = x[:, -1, :]  # last visit
        return {
            "predicted": self.predictor(last),
            "pattern_logits": self.classifier(last)
        }


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class SpirometryDataset(Dataset):
    """
    Expects:
    data_dir/
      patients.json     # {"patient_id": {"visits": [{"date": "...", "params": {...}, "pattern": 0-3, "age": ..., "sex": ...}], ...}}
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_seq_len: int = MAX_SEQ_LEN,
        scaler: Optional[StandardScaler] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_seq_len = max_seq_len

        labels_path = self.data_dir / "patients.json"
        if not labels_path.exists():
            raise FileNotFoundError(f"patients.json not found in {data_dir}")
        with open(labels_path) as f:
            self.patients = json.load(f)

        self.patient_ids = list(self.patients.keys())
        print(f"[{split}] Loaded {len(self.patient_ids)} patients")

        # Fit or use scaler
        if scaler is None:
            self.scaler = StandardScaler()
            # Collect all parameter vectors
            all_params = []
            for pid in self.patient_ids:
                for visit in self.patients[pid]["visits"]:
                    params = [visit["params"].get(p, 0.0) for p in PFT_PARAMETERS]
                    all_params.append(params)
            if all_params:
                self.scaler.fit(np.array(all_params))
        else:
            self.scaler = scaler

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict:
        pid = self.patient_ids[idx]
        patient = self.patients[pid]
        visits = patient["visits"]

        # Sort by date
        visits = sorted(visits, key=lambda v: v["date"])

        # Take last max_seq_len visits
        visits = visits[-self.max_seq_len:]
        seq_len = len(visits)

        # Build sequence
        seq = np.zeros((seq_len, INPUT_DIM), dtype=np.float32)
        for i, visit in enumerate(visits):
            params = [visit["params"].get(p, 0.0) for p in PFT_PARAMETERS]
            seq[i] = params

        # Scale
        seq = self.scaler.transform(seq)

        # Pattern label (from last visit)
        pattern = visits[-1].get("pattern", 0)

        # Padding mask for transformer (True = masked/padded)
        mask = None
        if seq_len < self.max_seq_len:
            # We'll pad in collate_fn
            pass

        return {
            "sequence": torch.from_numpy(seq),  # [seq_len, INPUT_DIM]
            "pattern": torch.tensor(pattern, dtype=torch.long),
            "patient_id": pid,
            "seq_len": seq_len,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Pad sequences to max length in batch."""
    max_len = max(item["seq_len"] for item in batch)
    batch_size = len(batch)

    padded_seq = torch.zeros(batch_size, max_len, INPUT_DIM, dtype=torch.float32)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)  # True = pad
    patterns = torch.zeros(batch_size, dtype=torch.long)
    patient_ids = []

    for i, item in enumerate(batch):
        seq_len = item["seq_len"]
        padded_seq[i, :seq_len] = item["sequence"]
        mask[i, :seq_len] = False
        patterns[i] = item["pattern"]
        patient_ids.append(item["patient_id"])

    return {
        "sequence": padded_seq,
        "mask": mask,
        "pattern": patterns,
        "patient_ids": patient_ids,
    }


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    accum_steps: int = 1,
) -> Dict:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for i, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        seq = batch['sequence'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        pattern = batch['pattern'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            out = model(seq, mask)
            cls_loss = F.cross_entropy(out['pattern_logits'], pattern)
            pred_loss = F.mse_loss(out['predicted'], seq[:, -1, :])
            loss = cls_loss + 0.1 * pred_loss
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum_steps
        pred = out['pattern_logits'].argmax(dim=1)
        correct += (pred == pattern).sum().item()
        total += pattern.size(0)

    return {
        "loss": total_loss / len(loader),
        "acc": correct / total,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        seq = batch['sequence'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        pattern = batch['pattern'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            out = model(seq, mask)
            cls_loss = F.cross_entropy(out['pattern_logits'], pattern)
            pred_loss = F.mse_loss(out['predicted'], seq[:, -1, :])
            loss = cls_loss + 0.1 * pred_loss

        total_loss += loss.item()
        pred = out['pattern_logits'].argmax(dim=1)
        correct += (pred == pattern).sum().item()
        total += pattern.size(0)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(pattern.cpu().numpy())

    from sklearn.metrics import classification_report
    report = classification_report(all_targets, all_preds, target_names=PATTERN_CLASSES, output_dict=True, zero_division=0)

    return {
        "loss": total_loss / len(loader),
        "acc": correct / total,
        "report": report,
    }


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scaler_obj: StandardScaler,
                   epoch: int, metrics: Dict, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_mean": scaler_obj.mean_.tolist() if hasattr(scaler_obj, 'mean_') else None,
        "scaler_std": np.sqrt(scaler_obj.var_).tolist() if hasattr(scaler_obj, 'var_') else None,
        "metrics": metrics,
    }, path)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train SpirometryTransformer for PFT pattern classification")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root (with patients.json)")
    parser.add_argument("--output-dir", type=str, default="python/models", help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Dataset
    full_ds = SpirometryDataset(args.data_dir, split="train")
    val_size = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    # Share scaler with val
    val_ds.dataset.scaler = full_ds.scaler

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )

    # Model
    model = SpirometryTransformer(
        input_dim=INPUT_DIM,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        num_patterns=NUM_PATTERNS,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler_amp = torch.amp.GradScaler('cuda')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}\n")

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_metrics = train_epoch(model, train_loader, optimizer, scaler_amp, device, args.accum_steps)
        print(f"Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['acc']:.4f}")

        val_metrics = validate(model, val_loader, device)
        print(f"Val:   loss={val_metrics['loss']:.4f}, acc={val_metrics['acc']:.4f}")
        for cls in PATTERN_CLASSES:
            if cls in val_metrics['report']:
                print(f"  {cls}: precision={val_metrics['report'][cls]['precision']:.3f}, recall={val_metrics['report'][cls]['recall']:.3f}, f1={val_metrics['report'][cls]['f1-score']:.3f}")

        scheduler.step()

        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            save_checkpoint(model, optimizer, full_ds.scaler, epoch, val_metrics, out_dir, "spirometry_best")

        save_checkpoint(model, optimizer, full_ds.scaler, epoch, val_metrics, out_dir, "spirometry_latest")

    print(f"\nTraining complete. Best val acc: {best_acc:.4f}")
    print(f"Checkpoints in: {out_dir}")


if __name__ == "__main__":
    main()