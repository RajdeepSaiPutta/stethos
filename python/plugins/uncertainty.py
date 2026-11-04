"""
Uncertainty Quantification & Conformal Prediction for all modalities.
Implements temperature scaling, conformal prediction sets, and calibration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of temperature scaling calibration."""
    temperature: float
    ece_before: float
    ece_after: float
    nce_before: float
    nce_after: float


@dataclass
class ConformalSet:
    """Conformal prediction set with coverage guarantee."""
    prediction_set: List[int]
    coverage: float
    set_size: float
    alpha: float


class TemperatureScaler(nn.Module):
    """Temperature scaling for neural network calibration."""

    def __init__(self, init_temp: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temp))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50, lr: float = 0.01) -> CalibrationResult:
        """Calibrate temperature on validation set."""
        self.eval()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_ece(logits, labels, temp):
            probs = F.softmax(logits / temp, dim=1)
            confidences, predictions = probs.max(1)
            accuracies = predictions.eq(labels)
            bins = torch.linspace(0, 1, 11)
            ece = 0.0
            for i in range(10):
                mask = (confidences >= bins[i]) & (confidences < bins[i+1])
                if mask.any():
                    bin_acc = accuracies[mask].float().mean()
                    bin_conf = confidences[mask].mean()
                    ece += mask.float().mean() * (bin_conf - bin_acc).abs()
            return ece

        def eval_nce(logits, labels, temp):
            """Negative log-likelihood calibration error."""
            probs = F.softmax(logits / temp, dim=1)
            true_probs = probs[torch.arange(len(labels)), labels]
            return -true_probs.log().mean()

        logits_orig = logits.detach()
        labels_orig = labels.detach()

        ece_before = eval_ece(logits_orig, labels_orig, 1.0)
        nce_before = eval_nce(logits_orig, labels_orig, 1.0)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits_orig / self.temperature, labels_orig)
            loss.backward()
            return loss

        optimizer.step(closure)

        ece_after = eval_ece(logits_orig, labels_orig, self.temperature.item())
        nce_after = eval_nce(logits_orig, labels_orig, self.temperature.item())

        return CalibrationResult(
            temperature=self.temperature.item(),
            ece_before=ece_before.item(),
            ece_after=ece_after.item(),
            nce_before=nce_before.item(),
            nce_after=nce_after.item()
        )


class ConformalPredictor:
    """Split conformal prediction for classification."""

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Target miscoverage rate (1 - alpha = coverage)
        """
        self.alpha = alpha
        self.threshold = None
        self.calibrated = False

    def calibrate(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """
        Calibrate conformal threshold on holdout set.

        Args:
            scores: [n_samples, n_classes] - softmax probabilities or confidence scores
            labels: [n_samples] - true labels
        Returns:
            threshold value
        """
        # Use 1 - max_prob as nonconformity score
        max_probs = scores.max(axis=1)
        nonconformity = 1 - max_probs

        # Conformal threshold
        n = len(nonconformity)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.threshold = np.quantile(nonconformity, q, interpolation='higher')
        self.calibrated = True
        return float(self.threshold)

    def predict_set(self, scores: np.ndarray) -> List[List[int]]:
        """
        Generate prediction sets for new samples.

        Args:
            scores: [n_samples, n_classes] - softmax probabilities
        Returns:
            List of prediction sets (lists of class indices)
        """
        if not self.calibrated:
            raise ValueError("Must call calibrate() first")

        max_probs = scores.max(axis=1)
        nonconformity = 1 - max_probs

        # Include all classes with nonconformity <= threshold
        prediction_sets = []
        for i in range(len(scores)):
            # Include classes where 1 - prob <= threshold  =>  prob >= 1 - threshold
            included = np.where(scores[i] >= 1 - self.threshold)[0].tolist()
            if not included:
                # Always include at least the top prediction
                included = [int(scores[i].argmax())]
            prediction_sets.append(included)

        return prediction_sets

    def evaluate_coverage(self, scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate coverage and average set size on test set.

        Returns:
            (coverage, avg_set_size)
        """
        sets = self.predict_set(scores)
        covered = sum(1 for i, s in enumerate(sets) if labels[i] in s)
        coverage = covered / len(labels)
        avg_size = sum(len(s) for s in sets) / len(sets)
        return coverage, avg_size


class UncertaintyQuantifier:
    """Unified uncertainty quantification for a modality."""

    def __init__(self, n_classes: int, alpha: float = 0.1):
        self.n_classes = n_classes
        self.conformal = ConformalPredictor(alpha=alpha)
        self.temperature_scaler = TemperatureScaler()
        self.calibrated = False

    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        """Calibrate both temperature scaling and conformal prediction."""
        # Temperature scaling
        cal_result = self.temperature_scaler.calibrate(logits, labels)

        # Get calibrated probabilities
        with torch.no_grad():
            cal_probs = F.softmax(logits / self.temperature_scaler.temperature, dim=1).cpu().numpy()

        # Conformal calibration
        self.conformal.calibrate(cal_probs, labels.cpu().numpy())
        self.calibrated = True

        return {
            "temperature": cal_result.temperature,
            "ece_before": cal_result.ece_before,
            "ece_after": cal_result.ece_after,
            "nce_before": cal_result.nce_before,
            "nce_after": cal_result.nce_after,
            "conformal_threshold": self.conformal.threshold,
            "alpha": self.conformal.alpha
        }

    def predict_with_uncertainty(self, logits: torch.Tensor) -> Dict[str, Any]:
        """
        Predict with uncertainty quantification.

        Returns:
            dict with: prediction, probabilities, prediction_set, conformal_threshold, temperature
        """
        if not self.calibrated:
            logger.warning("UQ not calibrated, using raw logits")

        with torch.no_grad():
            if self.calibrated:
                probs = F.softmax(logits / self.temperature_scaler.temperature, dim=1)
            else:
                probs = F.softmax(logits, dim=1)

        probs_np = probs.cpu().numpy()
        predictions = probs_np.argmax(axis=1)
        max_probs = probs_np.max(axis=1)

        # Conformal prediction sets
        if self.calibrated:
            pred_sets = self.conformal.predict_set(probs_np)
        else:
            pred_sets = [[int(p)] for p in predictions]

        return {
            "predictions": predictions.tolist(),
            "probabilities": probs_np.tolist(),
            "max_probabilities": max_probs.tolist(),
            "prediction_sets": pred_sets,
            "temperature": self.temperature_scaler.temperature.item() if self.calibrated else 1.0,
            "conformal_threshold": self.conformal.threshold if self.calibrated else None,
            "alpha": self.conformal.alpha if self.calibrated else 0.1
        }

    def save(self, path: str) -> None:
        """Save calibration state."""
        state = {
            "temperature": self.temperature_scaler.temperature.item(),
            "conformal_threshold": self.conformal.threshold,
            "alpha": self.conformal.alpha,
            "calibrated": self.calibrated
        }
        with open(path, 'w') as f:
            json.dump(state, f)

    def load(self, path: str) -> None:
        """Load calibration state."""
        with open(path) as f:
            state = json.load(f)
        self.temperature_scaler.temperature.data = torch.tensor(state["temperature"])
        self.conformal.threshold = state["conformal_threshold"]
        self.conformal.alpha = state["alpha"]
        self.conformal.calibrated = state["calibrated"]
        self.calibrated = state["calibrated"]


# Convenience functions for each modality
def create_breath_uq(alpha: float = 0.1) -> UncertaintyQuantifier:
    return UncertaintyQuantifier(n_classes=4, alpha=alpha)


def create_cxr_uq(alpha: float = 0.1) -> UncertaintyQuantifier:
    return UncertaintyQuantifier(n_classes=14, alpha=alpha)


def create_spirometry_uq(alpha: float = 0.1) -> UncertaintyQuantifier:
    return UncertaintyQuantifier(n_classes=4, alpha=alpha)  # pattern classification