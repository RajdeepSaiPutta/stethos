#!/usr/bin/env python3
"""
Breath Sound Model Training Script
Trains CNN on log-mel spectrograms for ICBHI-style classification (normal, crackle, wheeze, both).
Saves checkpoint to python/models/breath_cnn.pt
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
import torchaudio
import torchaudio.transforms as T
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

# ---------------------------------------------------------------------
# Model (mirror plugins/breath.py exactly)
# ---------------------------------------------------------------------

CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]
NUM_CLASSES = len(CLASS_NAMES)
TARGET_SR = 16000
N_MELS = 64
N_FFT = 400
HOP_LENGTH = 160
N_TIME = 128  # Fixed time dimension


class BreathCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class BreathDataset(Dataset):
    """
    Expects:
    data_dir/
      audio/              # .wav files
      labels.json         # {"filename.wav": {"label": 0-3, "patient_id": "..."}}
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        augments: bool = True,
        target_sr: int = TARGET_SR,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_time: int = N_TIME,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_time = n_time

        labels_path = self.data_dir / "labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.json not found in {data_dir}")
        with open(labels_path) as f:
            self.labels = json.load(f)

        self.audio_files = list(self.labels.keys())
        print(f"[{split}] Loaded {len(self.audio_files)} audio files")

        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.db_transform = T.AmplitudeToDB()

        # Augmentations (specaugment style)
        if augments and split == "train":
            self.freq_mask = T.FrequencyMasking(freq_mask_param=8)
            self.time_mask = T.TimeMasking(time_mask_param=10)
        else:
            self.freq_mask = None
            self.time_mask = None

    def _load_audio(self, fname: str) -> torch.Tensor:
        path = self.data_dir / "audio" / fname
        sr, audio = wavfile.read(str(path))
        # Convert to float
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
        # Resample if needed
        if sr != self.target_sr:
            import scipy.signal as signal
            audio = signal.resample(audio, int(len(audio) * self.target_sr / sr))
            sr = self.target_sr
        # Mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return torch.from_numpy(audio).unsqueeze(0)  # [1, samples]

    def _pad_or_crop(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: [1, n_mels, time]
        t = spec.shape[2]
        if t > self.n_time:
            start = (t - self.n_time) // 2
            spec = spec[:, :, start:start + self.n_time]
        elif t < self.n_time:
            pad = self.n_time - t
            spec = F.pad(spec, (0, pad), mode='constant', value=-80.0)  # -80dB ~ silence
        return spec

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Dict:
        fname = self.audio_files[idx]
        label = self.labels[fname]
        # label can be int or dict
        if isinstance(label, dict):
            target = label["label"]
        else:
            target = label

        waveform = self._load_audio(fname)
        spec = self.mel_transform(waveform)  # [1, n_mels, time]
        spec = self.db_transform(spec)  # [1, n_mels, time]
        spec = self._pad_or_crop(spec)

        if self.freq_mask is not None and self.split == "train":
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)

        return {
            "spectrogram": spec.squeeze(0),  # [n_mels, n_time]
            "label": torch.tensor(target, dtype=torch.long),
            "fname": fname,
        }


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for i, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        spec = batch['spectrogram'].to(device, non_blocking=True)  # [B, n_mels, n_time]
        target = batch['label'].to(device, non_blocking=True)

        spec = spec.unsqueeze(1)  # [B, 1, n_mels, n_time]

        with torch.amp.autocast('cuda'):
            logits = model(spec)
            loss = criterion(logits, target)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum_steps
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    return {
        "loss": total_loss / len(loader),
        "acc": correct / total,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        spec = batch['spectrogram'].to(device, non_blocking=True).unsqueeze(1)
        target = batch['label'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            logits = model(spec)
            loss = criterion(logits, target)

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    # Per-class accuracy
    from sklearn.metrics import classification_report
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    return {
        "loss": total_loss / len(loader),
        "acc": correct / total,
        "report": report,
    }


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train BreathCNN for respiratory sound classification")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root (with audio/, labels.json)")
    parser.add_argument("--output-dir", type=str, default="python/models", help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-time", type=int, default=N_TIME)
    parser.add_argument("--n-mels", type=int, default=N_MELS)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Dataset
    full_ds = BreathDataset(args.data_dir, split="train", augments=True, n_time=args.n_time, n_mels=args.n_mels)
    val_size = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    # Fresh val without augs
    val_ds_clean = BreathDataset(args.data_dir, split="val", augments=False, n_time=args.n_time, n_mels=args.n_mels)
    val_ds_clean.audio_files = [full_ds.audio_files[i] for i in val_ds.indices]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds_clean, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = BreathCNN(num_classes=NUM_CLASSES).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')
    criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds_clean)}\n")

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, criterion, device, args.accum_steps)
        print(f"Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['acc']:.4f}")

        val_metrics = validate(model, val_loader, criterion, device)
        print(f"Val:   loss={val_metrics['loss']:.4f}, acc={val_metrics['acc']:.4f}")
        for cls in CLASS_NAMES:
            if cls in val_metrics['report']:
                print(f"  {cls}: precision={val_metrics['report'][cls]['precision']:.3f}, recall={val_metrics['report'][cls]['recall']:.3f}, f1={val_metrics['report'][cls]['f1-score']:.3f}")

        scheduler.step()

        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            save_checkpoint(model, optimizer, epoch, val_metrics, out_dir, "breath_cnn_best")

        save_checkpoint(model, optimizer, epoch, val_metrics, out_dir, "breath_cnn_latest")

    print(f"\nTraining complete. Best val acc: {best_acc:.4f}")
    print(f"Checkpoints in: {out_dir}")


if __name__ == "__main__":
    main()