# stethos

StethOS - local-only multimodal respiratory diagnostics desktop app. breath sounds, chest x-ray, and spirometry go in; a fused risk assessment and a structured clinical report come out.

built with tauri (rust shell) + react (frontend) + a python fastapi sidecar that runs all inference. nothing leaves this machine - no cloud calls, no telemetry.

## architecture

- `src/` - react frontend. dashboard, per-modality panels, fusion view, report view, model manager, status bar.
- `src-tauri/` - rust shell. starts/stops the python sidecar, proxies inference calls to it.
- `python/plugins/` - inference plugins, one per modality:
  - `breath.py` - breath sound classification (normal / crackle / wheeze / both) on log-mel spectrograms
  - `cxr.py` - chest x-ray pathology classification (14 chexpert findings) + lung segmentation
  - `spirometry.py` - PFT pattern classification, GOLD staging, trajectory prediction
  - `fusion.py` - cross-attention fusion of all modalities into one risk assessment
- `python/api/main.py` - fastapi sidecar: `/health`, `/models`, `/inference/upload`, `/fusion`, `/export`

## status

scaffold stage. architecture end-to-end verified: plugins register, load (falling back to locally-initialized weights when no checkpoint exists), infer on synthetic inputs, and fuse into a risk assessment. checkpoints and training pipelines come next.

## run the sidecar alone

```bash
cd python
pip install -r requirements.txt
python -m uvicorn api.main:app --port 8765
```

## dev

```bash
npm install
npm run dev        # vite frontend only
npm run tauri:dev  # full desktop shell
```

research tool, not a medical device - confirm findings with a clinician.
