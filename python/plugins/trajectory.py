"""
Longitudinal Disease Trajectory Modeling for Spirometry.
Extends the SpirometryTransformer with trajectory prediction and disease progression.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPrediction:
    """Longitudinal trajectory prediction output."""
    predicted_visits: List[Dict[str, float]]  # Future visit predictions
    time_horizon: List[float]  # Time points (months from last visit)
    confidence_intervals: Dict[str, List[Tuple[float, float]]]  # 95% CI per parameter
    progression_risk: Dict[str, float]  # Risk of decline per parameter
    gold_progression: Optional[Dict[str, float]] = None  # GOLD stage progression risk


class TrajectoryTransformer(nn.Module):
    """Extended transformer for longitudinal trajectory prediction."""

    def __init__(
        self,
        input_dim: int = 11,  # PFT_PARAMETERS count
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 100,
        forecast_horizon: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Trajectory forecasting head
        self.trajectory_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, d_model * 4, dropout=dropout, batch_first=True),
            num_layers=2
        )

        self.forecast_proj = nn.Linear(d_model, input_dim)
        self.uncertainty_proj = nn.Linear(d_model, input_dim * 2)  # mean + log_var

        # GOLD stage progression
        self.gold_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5)  # 5 GOLD stages: None, 1, 2, 3, 4
        )

        # Progression risk heads
        self.risk_heads = nn.ModuleDict({
            'fev1_decline': nn.Linear(d_model, 1),
            'fvc_decline': nn.Linear(d_model, 1),
            'exacerbation': nn.Linear(d_model, 1),
        })

    def forward(
        self,
        x: torch.Tensor,
        time_deltas: Optional[torch.Tensor] = None,
        forecast_steps: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, input_dim] - historical PFT visits
            time_deltas: [batch, seq_len] - time between visits (months)
            forecast_steps: number of future visits to predict
        """
        batch_size, seq_len, _ = x.shape
        horizon = forecast_steps or self.forecast_horizon

        # Encode history
        x = self.input_proj(x) + self.pos_encoding[:, :seq_len, :]
        memory = self.encoder(x)  # [batch, seq_len, d_model]

        # Last visit embedding for risk prediction
        last_embedding = memory[:, -1, :]  # [batch, d_model]

        # Risk predictions
        risks = {}
        for name, head in self.risk_heads.items():
            risks[name] = torch.sigmoid(head(last_embedding))

        # GOLD progression
        gold_logits = self.gold_head(last_embedding)
        gold_probs = F.softmax(gold_logits, dim=1)

        # Forecast future visits
        forecast_steps = forecast_steps or self.forecast_horizon
        tgt = torch.zeros(batch_size, horizon, self.d_model, device=x.device)
        tgt = tgt + self.pos_encoding[:, seq_len:seq_len + horizon, :]

        forecast_emb = self.trajectory_decoder(tgt, memory)  # [batch, horizon, d_model]

        # Forecast parameters with uncertainty
        forecast_params = self.forecast_proj(forecast_emb)  # [batch, horizon, input_dim]
        uncertainty = self.uncertainty_proj(forecast_emb)  # [batch, horizon, input_dim * 2]
        mean = uncertainty[..., :self.input_dim]
        log_var = uncertainty[..., self.input_dim:]

        return {
            "forecast_mean": forecast_params,
            "forecast_log_var": log_var,
            "risks": risks,
            "gold_probs": gold_probs,
            "last_embedding": last_embedding,
            "memory": memory
        }


class LongitudinalSpirometryPlugin:
    """Wrapper for longitudinal spirometry analysis with trajectory modeling."""

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrajectoryTransformer().to(self.device)
        self.model.eval()

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded trajectory model from {model_path}")

    def predict_trajectory(
        self,
        historical_visits: List[Dict[str, float]],
        time_deltas: Optional[List[float]] = None,
        forecast_horizon: int = 5
    ) -> TrajectoryPrediction:
        """
        Predict future spirometry trajectory from historical visits.

        Args:
            historical_visits: List of visit dicts with PFT parameters
            time_deltas: Months between visits (optional)
            forecast_horizon: Number of future visits to predict
        Returns:
            TrajectoryPrediction with forecasts, CIs, and risk scores
        """
        from plugins.spirometry import PFT_PARAMETERS

        # Convert to tensor
        sequences = []
        for visit in historical_visits:
            seq = [visit.get(p, 0.0) for p in PFT_PARAMETERS]
            sequences.append(seq)

        arr = np.array(sequences, dtype=np.float32)
        x = torch.from_numpy(arr).unsqueeze(0).to(self.device)

        time_tensor = None
        if time_deltas:
            time_tensor = torch.tensor([time_deltas], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output = self.model(x, time_tensor, forecast_steps=forecast_horizon)

        # Extract forecasts
        forecast_mean = output["forecast_mean"][0].cpu().numpy()
        forecast_log_var = output["forecast_log_var"][0].cpu().numpy()

        # 95% CI: mean ± 1.96 * std
        forecast_std = np.exp(0.5 * forecast_log_var)
        ci_lower = forecast_mean - 1.96 * forecast_std
        ci_upper = forecast_mean + 1.96 * forecast_std

        # Build confidence intervals per parameter
        from plugins.spirometry import PFT_PARAMETERS
        ci_dict = {}
        for i, param in enumerate(PFT_PARAMETERS):
            ci_dict[param] = list(zip(ci_lower[:, i].tolist(), ci_upper[:, i].tolist()))

        # Risk scores
        risks = {k: float(v.item()) for k, v in output["risks"].items()}

        # GOLD progression
        gold_probs = F.softmax(output["gold_probs"], dim=1)[0].cpu().numpy()
        gold_dict = {f"GOLD_{i}": float(gold_probs[i]) for i in range(5)}  # 0=None, 1-4

        # Time horizon (assuming 6-month intervals)
        time_horizon = [(i + 1) * 6 for i in range(forecast_horizon)]

        return TrajectoryPrediction(
            predicted_visits=[dict(zip(PFT_PARAMETERS, row)) for row in forecast_mean],
            time_horizon=time_horizon,
            confidence_intervals=ci_dict,
            progression_risk=risks,
            gold_progression=gold_dict
        )

    def save_model(self, path: str) -> None:
        torch.save({"model_state_dict": self.model.state_dict()}, path)


def create_longitudinal_plugin(model_path: Optional[str] = None) -> LongitudinalSpirometryPlugin:
    """Factory function to create longitudinal plugin."""
    return LongitudinalSpirometryPlugin(model_path=model_path)