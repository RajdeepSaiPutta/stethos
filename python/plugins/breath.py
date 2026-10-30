"""
Breath Sound Classification Plugin
Classifies respiratory sounds: Normal, Crackle, Wheeze, Both
"""

from plugins.base import ModalityPlugin, ModalityMetadata, InferenceInput, InferenceOutput, registry
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]
TARGET_SR = 16000
N_MELS = 64
N_FFT = 400
HOP_LENGTH = 160


class BreathCNN(nn.Module):
    """Small CNN for log-mel spectrogram classification."""

    def __init__(self, num_classes: int = 4, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
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


class BreathPlugin(ModalityPlugin):
    """Breath sound classification using CNN on log-mel spectrograms."""

    def __init__(self, model_path: Optional[str] = None):
        self._model: Optional[BreathCNN] = None
        self._device: Optional[torch.device] = None
        self._model_path = model_path
        self._mel_transform = None
        self._db_transform = None

    @property
    def metadata(self) -> ModalityMetadata:
        return ModalityMetadata(
            name="breath",
            display_name="Breath Sound Analysis",
            version="1.0.0",
            description="Classifies respiratory sounds into normal, crackle, wheeze, or both using CNN on log-mel spectrograms",
            input_types=["audio/wav", "audio/x-wav"],
            output_types=["application/json"],
            gpu_memory_mb=500,
            supported_devices=["cuda", "cpu"],
            tags=["respiratory", "auscultation", "classification", "icbhi"]
        )

    def load_model(self, device: torch.device) -> None:
        self._device = device
        self._model = BreathCNN(num_classes=len(CLASS_NAMES)).to(device)
        self._model.eval()

        if self._model_path and Path(self._model_path).exists():
            checkpoint = torch.load(self._model_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
            self._model.load_state_dict(state_dict)
            logger.info(f"Loaded breath model from {self._model_path}")
        else:
            logger.warning("No breath model checkpoint found, using random weights")

        self._mel_transform = T.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        ).to(device)
        self._db_transform = T.AmplitudeToDB(top_db=80.0).to(device)

    def unload_model(self) -> None:
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
        self._mel_transform = None
        self._db_transform = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load and resample audio to target sample rate."""
        try:
            waveform, sr = torchaudio.load(path)
        except Exception as e:
            # newer torchaudio needs torchcodec to decode; fall back to scipy for wav files
            logger.warning("torchaudio.load failed (%s); falling back to scipy.wavfile", e)
            from scipy.io import wavfile
            sr, data = wavfile.read(path)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128.0) / 128.0
            else:
                data = data.astype(np.float32)
            if data.ndim > 1:
                data = data.mean(axis=1)
            waveform = torch.from_numpy(data).unsqueeze(0)
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    def _compute_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute normalized log-mel spectrogram."""
        spec = self._mel_transform(waveform)
        spec = self._db_transform(spec)
        # Normalize to [0, 1]
        spec = (spec + 80.0) / 80.0
        spec = spec.clamp(0.0, 1.0)
        return spec

    def preprocess(self, input_data: InferenceInput) -> torch.Tensor:
        waveform = self._load_audio(input_data.data_path).to(self._device)
        spec = self._compute_spectrogram(waveform)
        # Add batch dimension
        return spec.unsqueeze(0)

    def infer(self, preprocessed: torch.Tensor, options: Dict[str, Any]) -> torch.Tensor:
        with torch.no_grad():
            logits = self._model(preprocessed)
            probs = F.softmax(logits, dim=1)
        return probs

    def postprocess(self, raw_output: torch.Tensor, options: Dict[str, Any]) -> InferenceOutput:
        probs = raw_output[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))

        # Per-class confidence
        class_probs = {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}

        # Clinical interpretation
        interpretation = self._interpret(class_probs)

        return InferenceOutput(
            modality=self.metadata.name,
            success=True,
            results={
                "prediction": CLASS_NAMES[pred_idx],
                "probabilities": class_probs,
                "interpretation": interpretation,
                "class_names": CLASS_NAMES
            },
            metadata={
                "model": "BreathCNN",
                "sample_rate": TARGET_SR,
                "n_mels": N_MELS
            }
        )

    def _interpret(self, probs: Dict[str, float]) -> str:
        """Generate clinical interpretation."""
        pred = max(probs, key=probs.get)
        confidence = probs[pred]

        interpretations = {
            "normal": "Breath sounds within normal limits. No adventitious sounds detected.",
            "crackle": f"Crackles detected (confidence: {confidence:.1%}). Suggests alveolar/interstitial pathology. Consider CHF, pneumonia, or fibrosis.",
            "wheeze": f"Wheezes detected (confidence: {confidence:.1%}). Indicates airway obstruction. Consider asthma, COPD, or bronchiolitis.",
            "both": f"Both crackles and wheezes detected (confidence: {confidence:.1%}). Mixed obstructive and restrictive pattern. Urgent evaluation recommended."
        }
        return interpretations.get(pred, "Unable to interpret")


# Register plugin
def register_breath_plugin(model_path: Optional[str] = None) -> None:
    plugin = BreathPlugin(model_path=model_path)
    registry.register(plugin)