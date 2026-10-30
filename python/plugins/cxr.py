"""
Chest X-Ray Analysis Plugin
Segmentation (MedSAM) + Classification (CheXNet-style) + Measurements
"""

from plugins.base import ModalityPlugin, ModalityMetadata, InferenceInput, InferenceOutput, registry
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

# CheXNet pathologies (14 classes)
CHEXNET_PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]

LUNG_CLASSES = ["background", "left_lung", "right_lung", "heart", "trachea"]


class CXRClassifier(nn.Module):
    """DenseNet-121 based classifier for chest X-ray pathologies."""

    def __init__(self, num_classes: int = 14, pretrained: bool = True):
        super().__init__()
        from torchvision.models import densenet121, DenseNet121_Weights
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)
        self.features = backbone.features
        num_features = backbone.classifier.in_features
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
    """Lightweight MedSAM-style segmentation for lung/heart structures."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        # Encoder (MobileNetV3-Small for efficiency); imagenet weights only on opt-in (offline by default)
        import os
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        download = os.environ.get("RESPIRATORY_AI_PRETRAINED") == "1"
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if download else None
        backbone = mobilenet_v3_small(weights=weights)
        self.encoder = backbone.features

        # Decoder (U-Net style)
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
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.decoder(features)


class CXRPlugin(ModalityPlugin):
    """Chest X-ray analysis: classification + segmentation + measurements."""

    def __init__(
        self,
        classifier_path: Optional[str] = None,
        segmenter_path: Optional[str] = None
    ):
        self._classifier: Optional[CXRClassifier] = None
        self._segmenter: Optional[MedSAMSegmenter] = None
        self._device: Optional[torch.device] = None
        self._classifier_path = classifier_path
        self._segmenter_path = segmenter_path
        self._transform = None

    @property
    def metadata(self) -> ModalityMetadata:
        return ModalityMetadata(
            name="cxr",
            display_name="Chest X-Ray Analysis",
            version="1.0.0",
            description="Pathology classification (14 CheXNet labels), lung/heart segmentation, and cardiothoracic ratio measurement",
            input_types=["image/dicom", "image/png", "image/jpeg"],
            output_types=["application/json", "image/png"],
            gpu_memory_mb=2000,
            supported_devices=["cuda", "cpu"],
            tags=["radiology", "chest-xray", "classification", "segmentation", "chexnet"]
        )

    def load_model(self, device: torch.device) -> None:
        self._device = device
        # only pull imagenet weights when explicitly opted in; offline first
        download = os.environ.get("RESPIRATORY_AI_PRETRAINED") == "1"
        self._classifier = CXRClassifier(num_classes=len(CHEXNET_PATHOLOGIES), pretrained=download).to(device)
        self._segmenter = MedSAMSegmenter(num_classes=len(LUNG_CLASSES)).to(device)
        self._classifier.eval()
        self._segmenter.eval()

        if self._classifier_path and Path(self._classifier_path).exists():
            checkpoint = torch.load(self._classifier_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
            self._classifier.load_state_dict(state_dict)
            logger.info(f"Loaded CXR classifier from {self._classifier_path}")

        if self._segmenter_path and Path(self._segmenter_path).exists():
            checkpoint = torch.load(self._segmenter_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
            self._segmenter.load_state_dict(state_dict)
            logger.info(f"Loaded CXR segmenter from {self._segmenter_path}")

        if not self._classifier_path or not Path(self._classifier_path).exists():
            logger.warning(
                "No CXR classifier checkpoint found; classifier uses "
                + ("ImageNet weights" if os.environ.get("RESPIRATORY_AI_PRETRAINED") == "1" else "random weights (offline mode)")
            )

        # store just a flag; preprocessing does the math inline (torch.nn.Lambda was removed from torch)
        self._transform = lambda x: ((x.float() / 255.0 if x.max() > 1.0 else x.float()) - 0.485) / 0.229

    def unload_model(self) -> None:
        for attr in ['_classifier', '_segmenter']:
            if getattr(self, attr) is not None:
                getattr(self, attr).cpu()
                delattr(self, attr)
        self._transform = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_image(self, path: str) -> torch.Tensor:
        """Load DICOM, PNG, or JPEG and convert to tensor."""
        path = Path(path)
        if path.suffix.lower() in ['.dcm', '.dicom']:
            import pydicom
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
            # Normalize DICOM
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        else:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0

        # Resize to 512x512
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        # imagenet-style backbones expect 3 channels; replicate grayscale
        img = np.repeat(img[np.newaxis, ...], 3, axis=0)
        return torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)  # [1, 3, 512, 512]

    def preprocess(self, input_data: InferenceInput) -> torch.Tensor:
        img = self._load_image(input_data.data_path)
        img = self._transform(img)
        return img.to(self._device)

    def infer(self, preprocessed: torch.Tensor, options: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            # Classification
            cls_logits = self._classifier(preprocessed)
            cls_probs = F.sigmoid(cls_logits)

            # Segmentation
            seg_logits = self._segmenter(preprocessed)
            seg_probs = F.softmax(seg_logits, dim=1)

        return {
            "classification": cls_probs,
            "segmentation": seg_probs
        }

    def _compute_measurements(self, seg_mask: np.ndarray) -> Dict[str, float]:
        """Compute cardiothoracic ratio and lung areas."""
        h, w = seg_mask.shape
        measurements = {}

        # Cardiothoracic ratio (CTR)
        heart_mask = (seg_mask == 3).astype(np.uint8)  # heart class
        left_lung = (seg_mask == 1).astype(np.uint8)
        right_lung = (seg_mask == 2).astype(np.uint8)

        if heart_mask.sum() > 0:
            heart_contours, _ = cv2.findContours(heart_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if heart_contours:
                heart_cnt = max(heart_contours, key=cv2.contourArea)
                heart_bbox = cv2.boundingRect(heart_cnt)
                heart_width = heart_bbox[2]

            lung_contours, _ = cv2.findContours(
                (left_lung | right_lung).astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if lung_contours:
                lung_cnt = max(lung_contours, key=cv2.contourArea)
                lung_bbox = cv2.boundingRect(lung_cnt)
                lung_width = lung_bbox[2]

                if lung_width > 0:
                    measurements["cardiothoracic_ratio"] = heart_width / lung_width

        # Lung areas
        measurements["left_lung_area_px"] = float(left_lung.sum())
        measurements["right_lung_area_px"] = float(right_lung.sum())
        measurements["heart_area_px"] = float(heart_mask.sum())

        return measurements

    def postprocess(self, raw_output: Dict[str, torch.Tensor], options: Dict[str, Any]) -> InferenceOutput:
        cls_probs = raw_output["classification"][0].cpu().numpy()
        seg_probs = raw_output["segmentation"][0].cpu().numpy()
        seg_mask = np.argmax(seg_probs, axis=0).astype(np.uint8)

        # Classification results
        pathology_probs = {name: float(prob) for name, prob in zip(CHEXNET_PATHOLOGIES, cls_probs)}
        positive_pathologies = [name for name, prob in pathology_probs.items() if prob > 0.5]

        # Measurements
        measurements = self._compute_measurements(seg_mask)

        # Segmentation visualization (encode as base64 for frontend)
        import base64
        seg_vis = self._create_segmentation_overlay(seg_mask)
        _, buffer = cv2.imencode('.png', seg_vis)
        seg_b64 = base64.b64encode(buffer).decode('utf-8')

        return InferenceOutput(
            modality=self.metadata.name,
            success=True,
            results={
                "pathologies": pathology_probs,
                "positive_findings": positive_pathologies,
                "measurements": measurements,
                "segmentation_classes": LUNG_CLASSES,
                "segmentation_overlay": f"data:image/png;base64,{seg_b64}"
            },
            metadata={
                "classifier": "DenseNet121",
                "segmenter": "MobileNetV3-Small U-Net",
                "input_resolution": "512x512"
            }
        )

    def _create_segmentation_overlay(self, seg_mask: np.ndarray) -> np.ndarray:
        """Create RGB visualization of segmentation."""
        colors = {
            0: [0, 0, 0],        # background
            1: [0, 255, 0],      # left lung - green
            2: [0, 200, 0],      # right lung - dark green
            3: [255, 0, 0],      # heart - red
            4: [255, 255, 0],    # trachea - yellow
        }
        vis = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
        for cls, color in colors.items():
            vis[seg_mask == cls] = color
        return vis


def register_cxr_plugin(
    classifier_path: Optional[str] = None,
    segmenter_path: Optional[str] = None
) -> None:
    plugin = CXRPlugin(
        classifier_path=classifier_path,
        segmenter_path=segmenter_path
    )
    registry.register(plugin)