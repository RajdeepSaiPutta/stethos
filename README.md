# stethos

a desktop app for respiratory diagnostics. drop in a breath recording, chest x-ray, or spirometry file and it classifies each one, then fuses the findings into a single risk assessment with a per-modality breakdown. reports export as pdf, fhir, dicom-sr, or json.

all inference runs locally in a python sidecar. uses cuda when a gpu is present, falls back to cpu otherwise — nothing ever leaves your machine. where no trained checkpoint exists yet, models fall back to locally-initialized weights so the whole pipeline stays runnable end-to-end.

built with tauri 2, react, and fastapi.

## install

download the installer for your platform:

- **github releases (cpu, all platforms):** https://github.com/RajdeepSaiPutta/stethos/releases/tag/v0.1.0
- **hugging face (cpu + cuda, full set):** https://huggingface.co/rajdeepp11/stethos

| platform | installer |
|----------|-----------|
| windows | `Stethos_<version>_x64-setup.exe` |
| macos (apple silicon) | `Stethos_<version>_aarch64.dmg` |
| debian / ubuntu | `Stethos_<version>_amd64.deb` |
| fedora / opensuse | `Stethos-<version>-1.x86_64.rpm` |
| arch / cachyos | `Stethos-<version>-1-x86_64.pkg.tar.zst` |
| any linux | `install-linux.sh` + AppImage |
| linux + nvidia gpu | the cuda set under `cuda/linux/` (hugging face) |

each installer registers stethos in your app menu / start menu / launchpad and bundles the inference engine + model weights — no terminal step, no separate download. see `release/README.md` for checksums, gpu notes, and uninstall instructions.

## run from source

```bash
npm ci
node scripts/build-installers.mjs                # build installer for this os
```

or dev mode:

```bash
# terminal 1 — inference sidecar
cd python && python -m uvicorn api.main:app --port 8765

# terminal 2 — desktop shell
npm run tauri:dev
```

### gpu acceleration

the packaged sidecar detects cuda at runtime. cpu-only pytorch wheel runs everything but reports `"device": "cpu"`. to produce a cuda-bundled installer, build the sidecar with a cuda torch install:

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
node scripts/build-installers.mjs
```

`curl -s http://127.0.0.1:8765/health` reports the device the shipped build picked.

## brand assets

all logos, favicons, pwa icons, and os icons generated from one source image:

```bash
python scripts/generate-brand-assets.py "/path/to/stethos logo.png"
npx tauri icon src-tauri/brand/source.png
---

research tool, not a medical device — confirm findings with a clinician.