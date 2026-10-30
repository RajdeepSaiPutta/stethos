# stethos

stethos is a desktop app for respiratory diagnostics. it takes breath sounds, a chest x-ray, or spirometry data, classifies each one, and fuses the separate findings into a single risk assessment with a per-modality breakdown. reports export as pdf, fhir, dicom-sr, or json.

all inference runs locally in a python sidecar - it uses cuda when a gpu is present and falls back to cpu otherwise, and nothing ever leaves the machine. where no trained checkpoint exists yet, the models fall back to locally-initialized weights so the whole pipeline stays runnable end to end.

built with tauri, react, and fastapi. run the sidecar with `python -m uvicorn api.main:app --port 8765` from `python/`, then `npm run tauri:dev` for the desktop shell.

research tool, not a medical device - confirm findings with a clinician.
