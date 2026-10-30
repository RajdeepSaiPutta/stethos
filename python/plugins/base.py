"""
Plugin architecture for multimodal respiratory AI modalities.
Each modality is a self-contained plugin with its own model, preprocessing, and inference.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import torch
import numpy as np
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class ModalityMetadata:
    """Metadata describing a modality plugin."""
    name: str
    display_name: str
    version: str
    description: str
    input_types: List[str]  # e.g., ["audio/wav", "image/dicom", "application/fhir+json"]
    output_types: List[str]
    gpu_memory_mb: int
    supported_devices: List[str] = field(default_factory=lambda: ["cuda", "cpu"])
    tags: List[str] = field(default_factory=list)


class InferenceInput(BaseModel):
    """Standardized input for any modality inference."""
    modality: str
    data_path: str
    options: Dict[str, Any] = Field(default_factory=dict)
    patient_id: Optional[str] = None
    study_id: Optional[str] = None


class InferenceOutput(BaseModel):
    """Standardized output from any modality inference."""
    modality: str
    success: bool
    results: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    inference_time_ms: float = 0.0
    gpu_memory_used_mb: float = 0.0


class FusionInput(BaseModel):
    """Input for cross-modality fusion inference."""
    modalities: Dict[str, InferenceOutput]  # modality_name -> inference output
    patient_context: Optional[Dict[str, Any]] = None
    fusion_options: Dict[str, Any] = Field(default_factory=dict)


class FusionOutput(BaseModel):
    """Output from cross-modality fusion."""
    success: bool
    fused_results: Dict[str, Any] = Field(default_factory=dict)
    modality_contributions: Dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0
    explanation: Optional[str] = None
    error: Optional[str] = None


class ModalityPlugin(ABC):
    """Base class for all modality plugins."""

    @property
    @abstractmethod
    def metadata(self) -> ModalityMetadata:
        """Return plugin metadata."""
        pass

    @abstractmethod
    def load_model(self, device: torch.device) -> None:
        """Load model weights onto device."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload model and free memory."""
        pass

    @abstractmethod
    def preprocess(self, input_data: InferenceInput) -> Any:
        """Preprocess input data for inference."""
        pass

    @abstractmethod
    def infer(self, preprocessed: Any, options: Dict[str, Any]) -> InferenceOutput:
        """Run inference on preprocessed data."""
        pass

    @abstractmethod
    def postprocess(self, raw_output: Any, options: Dict[str, Any]) -> InferenceOutput:
        """Postprocess raw model output into standardized format."""
        pass

    def run_inference(self, input_data: InferenceInput) -> InferenceOutput:
        """Full inference pipeline."""
        import time
        start = time.time()
        try:
            preprocessed = self.preprocess(input_data)
            raw_output = self.infer(preprocessed, input_data.options)
            output = self.postprocess(raw_output, input_data.options)
            output.inference_time_ms = (time.time() - start) * 1000
            return output
        except Exception as e:
            return InferenceOutput(
                modality=self.metadata.name,
                success=False,
                error=str(e),
                inference_time_ms=(time.time() - start) * 1000
            )

    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        return 0.0


class PluginRegistry:
    """Registry for managing modality plugins."""

    def __init__(self):
        self._plugins: Dict[str, ModalityPlugin] = {}
        self._loaded: Dict[str, bool] = {}

    def register(self, plugin: ModalityPlugin) -> None:
        """Register a plugin."""
        self._plugins[plugin.metadata.name] = plugin
        self._loaded[plugin.metadata.name] = False

    def get(self, name: str) -> Optional[ModalityPlugin]:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> List[ModalityMetadata]:
        """List all registered plugins."""
        return [p.metadata for p in self._plugins.values()]

    def load(self, name: str, device: torch.device) -> bool:
        """Load a plugin's model."""
        plugin = self._plugins.get(name)
        if not plugin:
            return False
        if self._loaded.get(name, False):
            return True
        try:
            plugin.load_model(device)
            self._loaded[name] = True
            return True
        except Exception:
            logger.exception("Failed to load plugin %s", name)
            return False


    def unload(self, name: str) -> bool:
        """Unload a plugin's model."""
        plugin = self._plugins.get(name)
        if not plugin:
            return False
        try:
            plugin.unload_model()
            self._loaded[name] = False
            return True
        except Exception:
            return False

    def is_loaded(self, name: str) -> bool:
        """Check if a plugin is loaded."""
        return self._loaded.get(name, False)


# Global registry instance
registry = PluginRegistry()