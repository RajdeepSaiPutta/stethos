"""
Cross-Modality Fusion Plugin
Combines breath sounds, chest X-ray, spirometry, and clinical data
"""

from plugins.base import ModalityPlugin, ModalityMetadata, InferenceInput, InferenceOutput, FusionInput, FusionOutput, registry
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion of multiple modalities."""

    def __init__(
        self,
        modality_dims: Dict[str, int],
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        num_classes: int = 5  # risk levels
    ):
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.d_model = d_model

        # Project each modality to common dimension
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, d_model) for name, dim in modality_dims.items()
        })

        # Modality embeddings
        self.modality_embeddings = nn.Parameter(
            torch.randn(len(self.modality_names), d_model)
        )

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

        # Output heads
        self.risk_head = nn.Sequential(
            nn.Linear(d_model * len(self.modality_names), 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        self.explanation_head = nn.Sequential(
            nn.Linear(d_model * len(self.modality_names), 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # embedding for explanation generation
        )

        self.contribution_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, modalities: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        modalities: {modality_name: [batch, seq_len, feat_dim] or [batch, feat_dim]}
        """
        batch_size = next(iter(modalities.values())).shape[0]
        device = next(iter(modalities.values())).device

        # Project and add modality embeddings; absent modalities are zero-padded so
        # the flattened risk head always sees one token per expected modality slot
        projected = []
        for i, name in enumerate(self.modality_names):
            if name not in modalities:
                projected.append(torch.zeros(batch_size, 1, self.d_model, device=device))
                continue
            x = modalities[name]
            if x.dim() == 2:
                x = x.unsqueeze(1)  # [batch, 1, feat]
            x = self.projections[name](x)  # [batch, seq_len, d_model]
            x = x + self.modality_embeddings[i].unsqueeze(0).unsqueeze(0)
            projected.append(x)

        if not projected:
            raise ValueError("No valid modalities provided")

        # Concatenate all modalities
        fused = torch.cat(projected, dim=1)  # [batch, total_seq, d_model]

        # Cross-attention fusion
        for attn, ln in zip(self.cross_attn_layers, self.layer_norms):
            attn_out, _ = attn(fused, fused, fused)
            fused = ln(fused + attn_out)

        # Pool across sequence
        pooled = fused.mean(dim=1)  # [batch, d_model]

        # Per-modality contributions (attention weights)
        contributions = {}
        for i, name in enumerate(self.modality_names):
            if name in modalities:
                # Use the first token of each modality
                mod_tokens = fused[:, i:i+1, :] if i < fused.shape[1] else fused[:, 0:1, :]
                contributions[name] = self.contribution_head(mod_tokens).squeeze(-1)

        # Risk prediction
        fused_flat = fused.reshape(batch_size, -1)
        risk_logits = self.risk_head(fused_flat)
        risk_probs = F.softmax(risk_logits, dim=1)

        # Explanation embedding
        explanation_emb = self.explanation_head(fused_flat)

        return {
            "risk_probs": risk_probs,
            "contributions": contributions,
            "explanation_embedding": explanation_emb,
            "fused_features": pooled
        }


class ExplanationGenerator(nn.Module):
    """Generates natural language explanations from fusion embeddings."""

    def __init__(self, emb_dim: int = 64, vocab_size: int = 5000, max_len: int = 100):
        super().__init__()
        self.emb_proj = nn.Linear(emb_dim, 256)
        self.decoder = nn.GRU(256, 256, batch_first=True)
        self.output = nn.Linear(256, vocab_size)
        self.max_len = max_len

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # Simple version: return embedding for template-based explanation
        return self.emb_proj(emb)


class FusionPlugin(ModalityPlugin):
    """Cross-modality fusion for comprehensive respiratory assessment."""

    def __init__(self, model_path: Optional[str] = None):
        self._model: Optional[CrossAttentionFusion] = None
        self._device: Optional[torch.device] = None
        self._model_path = model_path

        # Expected modality feature dimensions (from their respective plugins)
        self.modality_dims = {
            "breath": 4,       # class probabilities
            "cxr": 14,         # pathology probabilities
            "spirometry": 11,  # PFT parameters (count matches run_fusion feature list)
            "clinical": 20,    # age, sex, BMI, comorbidities, meds, etc.
        }

    @property
    def metadata(self) -> ModalityMetadata:
        return ModalityMetadata(
            name="fusion",
            display_name="Multimodal Fusion",
            version="1.0.0",
            description="Cross-attention fusion of breath sounds, chest X-ray, spirometry, and clinical data for unified risk assessment",
            input_types=["application/json"],
            output_types=["application/json"],
            gpu_memory_mb=1000,
            supported_devices=["cuda", "cpu"],
            tags=["fusion", "multimodal", "risk-assessment", "clinical-decision-support"]
        )

    def load_model(self, device: torch.device) -> None:
        self._device = device
        self._model = CrossAttentionFusion(self.modality_dims).to(device)
        self._model.eval()

        if self._model_path and Path(self._model_path).exists():
            checkpoint = torch.load(self._model_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
            self._model.load_state_dict(state_dict)
            logger.info(f"Loaded fusion model from {self._model_path}")
        else:
            logger.warning("No fusion model checkpoint found, using random weights")

    def unload_model(self) -> None:
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def preprocess(self, input_data: InferenceInput) -> Dict[str, torch.Tensor]:
        """Expects fusion input with modality results."""
        # This is called via run_fusion_inference, not directly
        return {}

    def infer(self, preprocessed: Any, options: Dict[str, Any]) -> Any:
        pass

    def postprocess(self, raw_output: Any, options: Dict[str, Any]) -> InferenceOutput:
        pass

    def run_fusion(self, fusion_input: FusionInput) -> FusionOutput:
        """Run fusion inference on multiple modality outputs."""
        import time
        start = time.time()

        try:
            # Extract features from each modality output
            modality_features = {}

            for mod_name, mod_output in fusion_input.modalities.items():
                if not mod_output.success:
                    continue

                results = mod_output.results
                if mod_name == "breath":
                    # Breath: use class probabilities
                    feats = [results["probabilities"].get(c, 0) for c in ["normal", "crackle", "wheeze", "both"]]
                elif mod_name == "cxr":
                    # CXR: use pathology probabilities
                    feats = [results["pathologies"].get(p, 0) for p in [
                        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
                        "Effusion", "Emphysema", "Fibrosis", "Hernia",
                        "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
                        "Pneumonia", "Pneumothorax"
                    ]]
                elif mod_name == "spirometry":
                    # Spirometry: use predicted values
                    feats = [results["predicted_values"].get(p, 0) for p in [
                        "FVC", "FEV1", "FEV1_FVC", "FEF25_75", "PEF",
                        "FIVC", "FIF50", "MVV", "TLC", "RV", "DLCO"
                    ]]
                elif mod_name == "clinical":
                    # Clinical: use provided features
                    feats = fusion_input.patient_context.get("features", [0]*20) if fusion_input.patient_context else [0]*20
                else:
                    continue

                modality_features[mod_name] = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(self._device)

            if not modality_features:
                return FusionOutput(
                    success=False,
                    error="No valid modality inputs provided"
                )

            # Run fusion
            with torch.no_grad():
                output = self._model(modality_features)

            risk_probs = output["risk_probs"][0].cpu().numpy()
            risk_levels = ["Very Low", "Low", "Moderate", "High", "Very High"]
            risk_idx = int(np.argmax(risk_probs))

            contributions = {k: float(v.item()) for k, v in output["contributions"].items()}

            # Generate explanation
            explanation = self._generate_explanation(
                risk_levels[risk_idx],
                contributions,
                fusion_input.modalities
            )

            return FusionOutput(
                success=True,
                fused_results={
                    "risk_level": risk_levels[risk_idx],
                    "risk_probabilities": {lvl: float(p) for lvl, p in zip(risk_levels, risk_probs)},
                    "risk_score": float(risk_idx) / (len(risk_levels) - 1)
                },
                modality_contributions=contributions,
                confidence=float(np.max(risk_probs)),
                explanation=explanation
            )

        except Exception as e:
            logger.error(f"Fusion inference failed: {e}")
            return FusionOutput(
                success=False,
                error=str(e)
            )

    def _generate_explanation(
        self,
        risk_level: str,
        contributions: Dict[str, float],
        modalities: Dict[str, Any]
    ) -> str:
        """Generate clinical explanation."""
        # Sort modalities by contribution
        sorted_mods = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        top_mod = sorted_mods[0][0] if sorted_mods else "unknown"

        mod_descriptions = {
            "breath": "auscultation findings",
            "cxr": "chest X-ray findings",
            "spirometry": "pulmonary function tests",
            "clinical": "clinical history"
        }

        explanation = f"Overall risk: {risk_level}. "

        if risk_level in ["High", "Very High"]:
            explanation += "Urgent clinical evaluation recommended. "
        elif risk_level == "Moderate":
            explanation += "Close monitoring and therapy optimization advised. "
        else:
            explanation += "Continue current management with routine follow-up. "

        explanation += f"Primary driver: {mod_descriptions.get(top_mod, top_mod)} "

        # Add specific findings
        findings = []
        for mod_name, mod_output in modalities.items():
            if not mod_output.success:
                continue
            results = mod_output.results
            if mod_name == "breath" and results.get("prediction") != "normal":
                findings.append(f"breath: {results['prediction']}")
            elif mod_name == "cxr" and results.get("positive_findings"):
                findings.append(f"CXR: {', '.join(results['positive_findings'][:3])}")
            elif mod_name == "spirometry" and results.get("pattern") != "Normal":
                findings.append(f"spirometry: {results['pattern']} pattern")

        if findings:
            explanation += f"Key findings: {'; '.join(findings)}."

        return explanation


def register_fusion_plugin(model_path: Optional[str] = None) -> None:
    plugin = FusionPlugin(model_path=model_path)
    registry.register(plugin)