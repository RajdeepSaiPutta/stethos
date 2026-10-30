"""
Spirometry / Pulmonary Function Test Plugin
Analyzes FEV1, FVC, FEV1/FVC ratio, flow-volume loops
"""

from plugins.base import ModalityPlugin, ModalityMetadata, InferenceInput, InferenceOutput, registry
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

PFT_PARAMETERS = [
    "FVC", "FEV1", "FEV1_FVC", "FEF25_75", "PEF",
    "FIVC", "FIF50", "MVV", "TLC", "RV", "DLCO"
]


class SpirometryTransformer(nn.Module):
    """Transformer for longitudinal spirometry analysis."""

    def __init__(self, input_dim: int = len(PFT_PARAMETERS), d_model: int = 128, nhead: int = 4, num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 50, d_model))  # max 50 visits
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, len(PFT_PARAMETERS))
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # Normal, Obstructive, Restrictive, Mixed
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [batch, seq_len, input_dim]
        seq_len = x.shape[1]
        x = self.input_proj(x) + self.pos_encoding[:, :seq_len, :]
        x = self.encoder(x)
        last = x[:, -1, :]  # last visit
        return {
            "predicted": self.predictor(last),
            "pattern_logits": self.classifier(last)
        }


class SpirometryPlugin(ModalityPlugin):
    """Spirometry analysis: pattern classification, trend prediction, GOLD staging."""

    def __init__(self, model_path: Optional[str] = None):
        self._model: Optional[SpirometryTransformer] = None
        self._device: Optional[torch.device] = None
        self._model_path = model_path
        self._scaler_mean = None
        self._scaler_std = None

    @property
    def metadata(self) -> ModalityMetadata:
        return ModalityMetadata(
            name="spirometry",
            display_name="Spirometry Analysis",
            version="1.0.0",
            description="Pulmonary function test interpretation: obstructive/restrictive pattern, GOLD/GLI staging, longitudinal trajectory",
            input_types=["application/json", "text/csv"],
            output_types=["application/json"],
            gpu_memory_mb=300,
            supported_devices=["cuda", "cpu"],
            tags=["pulmonology", "spirometry", "pft", "gold", "gli"]
        )

    def load_model(self, device: torch.device) -> None:
        self._device = device
        self._model = SpirometryTransformer().to(device)
        self._model.eval()

        if self._model_path and Path(self._model_path).exists():
            checkpoint = torch.load(self._model_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
            self._model.load_state_dict(state_dict)
            self._scaler_mean = checkpoint.get("scaler_mean")
            self._scaler_std = checkpoint.get("scaler_std")
            logger.info(f"Loaded spirometry model from {self._model_path}")
        else:
            logger.warning("No spirometry model checkpoint found")

    def unload_model(self) -> None:
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _parse_input(self, path: str) -> List[Dict[str, float]]:
        """Parse JSON or CSV spirometry data."""
        path = Path(path)
        if path.suffix.lower() == '.json':
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            return data
        elif path.suffix.lower() == '.csv':
            import pandas as pd
            df = pd.read_csv(path)
            return df.to_dict('records')
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        if self._scaler_mean is not None and self._scaler_std is not None:
            return (data - self._scaler_mean) / (self._scaler_std + 1e-8)
        return data

    def preprocess(self, input_data: InferenceInput) -> torch.Tensor:
        records = self._parse_input(input_data.data_path)
        # Extract parameters in order
        sequences = []
        for r in records:
            seq = [r.get(p, 0.0) for p in PFT_PARAMETERS]
            sequences.append(seq)
        arr = np.array(sequences, dtype=np.float32)
        arr = self._normalize(arr)
        return torch.from_numpy(arr).unsqueeze(0).to(self._device)  # [1, seq_len, dim]

    def infer(self, preprocessed: torch.Tensor, options: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            output = self._model(preprocessed)
        return output

    def postprocess(self, raw_output: Dict[str, torch.Tensor], options: Dict[str, Any]) -> InferenceOutput:
        predicted = raw_output["predicted"][0].cpu().numpy()
        pattern_logits = raw_output["pattern_logits"][0].cpu().numpy()
        pattern_probs = torch.softmax(torch.from_numpy(pattern_logits), dim=0).numpy()

        patterns = ["Normal", "Obstructive", "Restrictive", "Mixed"]
        pattern = patterns[int(np.argmax(pattern_probs))]

        # GOLD staging if obstructive
        gold_stage = None
        fev1_pred = predicted[1]  # FEV1 index
        if pattern in ["Obstructive", "Mixed"] and fev1_pred > 0:
            if fev1_pred >= 0.8: gold_stage = "GOLD 1 (Mild)"
            elif fev1_pred >= 0.5: gold_stage = "GOLD 2 (Moderate)"
            elif fev1_pred >= 0.3: gold_stage = "GOLD 3 (Severe)"
            else: gold_stage = "GOLD 4 (Very Severe)"

        # Predicted values
        predicted_dict = {p: float(v) for p, v in zip(PFT_PARAMETERS, predicted)}

        # Trend analysis (if multiple visits)
        trend = "stable"
        if hasattr(self, '_last_fev1') and self._last_fev1 is not None:
            if fev1_pred < self._last_fev1 * 0.95:
                trend = "declining"
            elif fev1_pred > self._last_fev1 * 1.05:
                trend = "improving"
        self._last_fev1 = fev1_pred

        return InferenceOutput(
            modality=self.metadata.name,
            success=True,
            results={
                "pattern": pattern,
                "pattern_probabilities": {p: float(v) for p, v in zip(patterns, pattern_probs)},
                "gold_stage": gold_stage,
                "predicted_values": predicted_dict,
                "trend": trend,
                "fev1_fvc_ratio": predicted_dict.get("FEV1_FVC", 0)
            },
            metadata={
                "model": "SpirometryTransformer",
                "parameters": PFT_PARAMETERS
            }
        )


def register_spirometry_plugin(model_path: Optional[str] = None) -> None:
    plugin = SpirometryPlugin(model_path=model_path)
    registry.register(plugin)