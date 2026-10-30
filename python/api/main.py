"""stethos sidecar
Local-only inference API for the stethos desktop shell.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import torch
import uvicorn
import logging
from pathlib import Path
import tempfile
import shutil
import json

# Import plugins
from plugins.base import registry, InferenceInput, InferenceOutput, FusionInput, FusionOutput
from plugins.breath import register_breath_plugin
from plugins.cxr import register_cxr_plugin
from plugins.spirometry import register_spirometry_plugin
from plugins.fusion import register_fusion_plugin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stethos Sidecar API",
    version="0.1.0",
    description="Local multimodal respiratory ML API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1420", "http://localhost:8765", "tauri://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device setup — accelerator first (cuda, then apple silicon mps), cpu as the
# floor. Plugins receive this device and move their weights onto it. A device
# request can pin this down to a specific accelerator ("cuda" | "mps") or to
# "cpu"; "auto" (or unset) takes the best available.
DEVICE_OVERRIDE: Optional[str] = None


def _pick_device(override: Optional[str] = None) -> torch.device:
    if override == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if override == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
    if override == "cpu":
        return torch.device("cpu")
    # auto: best available accelerator, cpu as the floor
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _pick_device(DEVICE_OVERRIDE)
logger.info(f"Using device: {DEVICE}")

# Model paths (relative to app data dir)
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Register all plugins
register_breath_plugin(str(MODEL_DIR / "breath_cnn.pt") if (MODEL_DIR / "breath_cnn.pt").exists() else None)
register_cxr_plugin(
    str(MODEL_DIR / "cxr_classifier.pt") if (MODEL_DIR / "cxr_classifier.pt").exists() else None,
    str(MODEL_DIR / "cxr_segmenter.pt") if (MODEL_DIR / "cxr_segmenter.pt").exists() else None
)
register_spirometry_plugin(str(MODEL_DIR / "spirometry.pt") if (MODEL_DIR / "spirometry.pt").exists() else None)
register_fusion_plugin(str(MODEL_DIR / "fusion.pt") if (MODEL_DIR / "fusion.pt").exists() else None)

# Load models on startup
@app.on_event("startup")
async def load_models():
    for name in list(registry._plugins):
        registry.load(name, DEVICE)

@app.on_event("shutdown")
async def unload_models():
    for name in registry._plugins:
        registry.unload(name)

# Request/Response models
class InferenceRequest(BaseModel):
    modality: str
    input_path: str
    options: Dict[str, Any] = {}
    patient_id: Optional[str] = None
    study_id: Optional[str] = None

class FusionRequest(BaseModel):
    modalities: Dict[str, Any]  # modality_name -> InferenceOutput dict
    patient_context: Optional[Dict[str, Any]] = None
    fusion_options: Dict[str, Any] = {}

class ExportRequest(BaseModel):
    format: str  # "pdf", "dicom-sr", "fhir", "json"
    data: Dict[str, Any]
    output_path: str

class ModelDownloadRequest(BaseModel):
    model: str

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "device_override": DEVICE_OVERRIDE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_total_mb": torch.cuda.get_device_properties(0).total_memory / (1024**2) if torch.cuda.is_available() else 0,
        "loaded_plugins": [name for name, loaded in registry._loaded.items() if loaded],
        "available_plugins": [p.name for p in registry.list_plugins()]
    }


class DeviceRequest(BaseModel):
    # "auto" | "cuda" | "mps" | "cpu" — re-loads every plugin on the new device
    device: str


@app.post("/device")
async def set_device(request: DeviceRequest):
    global DEVICE_OVERRIDE, DEVICE
    if request.device not in ("auto", "cuda", "mps", "cpu"):
        raise HTTPException(status_code=400, detail=f"unknown device: {request.device}")
    target = _pick_device(request.device if request.device != "auto" else None)
    if str(target) == str(DEVICE) and request.device == DEVICE_OVERRIDE:
        return await health_check()
    logger.info(f"Switching device {DEVICE} -> {target} (requested: {request.device})")
    for name in list(registry._plugins):
        registry.unload(name)
    DEVICE_OVERRIDE = request.device
    DEVICE = target
    for name in list(registry._plugins):
        registry.load(name, DEVICE)
    return await health_check()

# List available models
@app.get("/models")
async def list_models():
    return {
        "plugins": [
            {
                "name": p.name,
                "display_name": p.display_name,
                "version": p.version,
                "description": p.description,
                "input_types": p.input_types,
                "output_types": p.output_types,
                "gpu_memory_mb": p.gpu_memory_mb,
                "loaded": registry.is_loaded(p.name)
            }
            for p in registry.list_plugins()
        ]
    }

# Single modality inference
@app.post("/inference", response_model=Dict[str, Any])
async def run_inference(request: InferenceRequest):
    plugin = registry.get(request.modality)
    if not plugin:
        raise HTTPException(status_code=404, detail=f"Modality '{request.modality}' not found")

    if not registry.is_loaded(request.modality):
        success = registry.load(request.modality, DEVICE)
        if not success:
            raise HTTPException(status_code=503, detail=f"Failed to load {request.modality} model")

    input_data = InferenceInput(
        modality=request.modality,
        data_path=request.input_path,
        options=request.options,
        patient_id=request.patient_id,
        study_id=request.study_id,
    )
    output = plugin.run_inference(input_data)
    return output.dict()

# File upload + inference
@app.post("/inference/upload")
async def run_inference_upload(
    modality: str = Form(...),
    file: UploadFile = File(...),
    options: str = Form("{}"),
    patient_id: Optional[str] = Form(None),
    study_id: Optional[str] = Form(None)
):
    plugin = registry.get(modality)
    if not plugin:
        raise HTTPException(status_code=404, detail=f"Modality '{modality}' not found")

    if not registry.is_loaded(modality):
        success = registry.load(modality, DEVICE)
        if not success:
            raise HTTPException(status_code=503, detail=f"Failed to load {modality} model")

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        options_dict = json.loads(options)
        input_data = InferenceInput(
            modality=modality,
            data_path=tmp_path,
            options=options_dict,
            patient_id=patient_id,
            study_id=study_id
        )
        output = plugin.run_inference(input_data)
        return output.dict()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

# Fusion inference
@app.post("/fusion", response_model=Dict[str, Any])
async def run_fusion(request: FusionRequest):
    fusion_plugin = registry.get("fusion")
    if not fusion_plugin:
        raise HTTPException(status_code=404, detail="Fusion plugin not found")

    if not registry.is_loaded("fusion"):
        success = registry.load("fusion", DEVICE)
        if not success:
            raise HTTPException(status_code=503, detail="Failed to load fusion model")

    # Convert dict inputs to FusionInput
    modalities = {}
    for name, output_dict in request.modalities.items():
        try:
            modalities[name] = InferenceOutput(**output_dict)
        except Exception as e:
            logger.warning(f"Failed to parse modality {name}: {e}")

    fusion_input = FusionInput(
        modalities=modalities,
        patient_context=request.patient_context,
        fusion_options=request.fusion_options
    )

    output = fusion_plugin.run_fusion(fusion_input)
    return output.dict()

# Export report
@app.post("/export")
async def export_report(request: ExportRequest):
    # Delegate to Python export utilities
    from plugins.export import export_report
    try:
        path = export_report(request.format, request.data, request.output_path)
        return {"path": path, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Download model
@app.post("/models/download")
async def download_model(request: ModelDownloadRequest):
    model_name = request.model
    # Map model names to Hugging Face repos or URLs
    model_sources = {
        "breath_cnn": "rajdeepp11/respiratory-breath-cnn",
        "cxr_classifier": "rajdeepp11/respiratory-cxr-classifier",
        "cxr_segmenter": "rajdeepp11/respiratory-cxr-segmenter",
        "spirometry": "rajdeepp11/respiratory-spirometry",
        "fusion": "rajdeepp11/respiratory-fusion",
    }

    if model_name not in model_sources:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    try:
        from huggingface_hub import hf_hub_download
        repo_id = model_sources[model_name]
        local_path = hf_hub_download(repo_id=repo_id, filename=f"{model_name}.pt", local_dir=MODEL_DIR)
        # Reload the plugin
        plugin = registry.get(model_name.replace("_", "-"))  # adjust naming
        if plugin:
            plugin.load_model(DEVICE)
        return {"message": f"Downloaded {model_name} to {local_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")