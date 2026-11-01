#!/usr/bin/env python3
"""
Synthetic breath sound data generator for testing the training pipeline.
Creates fake audio files and labels.json in the expected format.
"""

import json
import numpy as np
from scipy.io import wavfile
from pathlib import Path
import argparse


CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]


def generate_synthetic_audio(
    out_path: Path,
    label: int,
    duration: float = 4.0,
    sr: int = 16000,
) -> None:
    """Generate synthetic respiratory sound for a given class."""
    t = np.arange(int(sr * duration)) / sr
    audio = np.zeros_like(t)

    # Base respiratory noise
    base = 0.02 * np.random.randn(len(t))

    if label == 0:  # normal - vesicular breath sounds
        # Low-frequency rumble (0.1-0.5 Hz "breathing" envelope) + broadband noise
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t)  # breathing cycle ~5s
        # Vesicular: soft, low-pitched, inspiratory longer than expiratory
        audio = env * (0.3 * np.sin(2 * np.pi * 150 * t) + base)

    elif label == 1:  # crackle - discontinuous, high-freq
        # Fine crackles: short bursts at 400-800 Hz
        for _ in range(int(duration * 3)):  # ~3 crackles per second
            start = np.random.randint(0, len(t) - int(0.01 * sr))
            dur = int(0.005 * sr)  # 5ms
            crackle_env = np.exp(-np.arange(dur) / (0.001 * sr))
            freq = np.random.uniform(400, 800)
            audio[start:start+dur] += 0.4 * crackle_env * np.sin(2 * np.pi * freq * np.arange(dur) / sr)
        audio += base

    elif label == 2:  # wheeze - continuous, musical, high-pitched
        # Wheeze: sustained sinusoid at 200-500 Hz with frequency modulation
        fm = 50 * np.sin(2 * np.pi * 0.5 * t)  # freq modulation
        carrier = np.sin(2 * np.pi * (300 + fm) * t)
        env = 0.5 + 0.3 * np.sin(2 * np.pi * 0.25 * t)  # breathing envelope
        audio = 0.4 * env * carrier + base

    elif label == 3:  # both - crackles + wheeze
        # Combine both
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t)
        # Wheeze component
        fm = 30 * np.sin(2 * np.pi * 0.5 * t)
        wheeze = 0.3 * env * np.sin(2 * np.pi * (250 + fm) * t)
        # Crackle component
        crackles = np.zeros_like(t)
        for _ in range(int(duration * 2)):
            start = np.random.randint(0, len(t) - int(0.01 * sr))
            dur = int(0.008 * sr)
            crackle_env = np.exp(-np.arange(dur) / (0.0015 * sr))
            freq = np.random.uniform(450, 750)
            crackles[start:start+dur] += 0.3 * crackle_env * np.sin(2 * np.pi * freq * np.arange(dur) / sr)
        audio = wheeze + crackles + base

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8
    audio = (audio * 32767).astype(np.int16)

    # Save
    wavfile.write(str(out_path), sr, audio)


def create_synthetic_dataset(out_dir: Path, n_per_class: int = 50, duration: float = 4.0):
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    labels = {}

    for class_idx, class_name in enumerate(CLASS_NAMES):
        for i in range(n_per_class):
            fname = f"{class_name}_{i:04d}.wav"
            out_path = audio_dir / fname
            generate_synthetic_audio(out_path, class_idx, duration=duration)
            labels[fname] = {"label": class_idx, "patient_id": f"synth_patient_{class_idx}_{i}"}

    with open(out_dir / "labels.json", 'w') as f:
        json.dump(labels, f, indent=2)

    total = len(labels)
    print(f"Created {total} synthetic samples in {out_dir}")
    print(f"  audio/: {len(list(audio_dir.glob('*.wav')))}")
    print(f"  labels.json: {total} entries")
    for c in CLASS_NAMES:
        count = sum(1 for v in labels.values() if v["label"] == CLASS_NAMES.index(c))
        print(f"    {c}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="data/breath_synth")
    parser.add_argument("--n-per-class", type=int, default=50)
    parser.add_argument("--duration", type=float, default=4.0)
    args = parser.parse_args()

    create_synthetic_dataset(Path(args.out_dir), args.n_per_class, args.duration)