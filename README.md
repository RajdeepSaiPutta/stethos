# Stethos

Stethos is a desktop app for respiratory diagnostics. It takes breath sounds, a
chest X-ray, or spirometry data, classifies each one, and fuses the separate
findings into a single risk assessment with a per-modality breakdown. Reports
export as PDF, FHIR, DICOM-SR, or JSON.

All inference runs locally in a Python sidecar. It uses CUDA when a GPU is
present and falls back to CPU otherwise; nothing ever leaves the machine. Where
no trained checkpoint exists yet, the models fall back to locally-initialized
weights so the whole pipeline stays runnable end to end.

Built with Tauri 2, React, and FastAPI.

## Install

Download the installer for your platform:

- **GitHub Releases (CPU, all platforms):**
  <https://github.com/RajdeepSaiPutta/stethos/releases/tag/v0.1.0>
- **Hugging Face (CPU + CUDA, full set):**
  <https://huggingface.co/rajdeepp11/stethos>

| Platform | Installer |
| --- | --- |
| Windows | `Stethos_<version>_x64-setup.exe` |
| macOS | `Stethos_<version>_aarch64.dmg` |
| Debian / Ubuntu | `Stethos_<version>_amd64.deb` |
| Fedora / openSUSE | `Stethos-<version>-1.x86_64.rpm` |
| Arch / CachyOS | `Stethos-<version>-1-x86_64.pkg.tar.zst` |
| Any Linux | `install-linux.sh` + AppImage |
| Linux + NVIDIA GPU | the CUDA set under `cuda/linux/` (Hugging Face) |

Each installer registers Stethos in the application menu / Start menu /
Launchpad and bundles the inference engine plus the model weights, so there is
no terminal step and no separate download. See [`release/README.md`](release/README.md)
for checksums, GPU notes, and uninstall instructions. The local `release/`
folders hold the same files on a build machine.

## Build from source

```bash
npm ci
node scripts/build-installers.mjs                # installer for this OS
```

Or run it in development:

```bash
# terminal 1 - inference sidecar
cd python && python -m uvicorn api.main:app --port 8765

# terminal 2 - desktop shell
npm run tauri:dev
```

### GPU acceleration

The packaged sidecar detects CUDA at runtime. A CPU-only PyTorch wheel still
runs everything, but reports `"device": "cpu"`. To produce a CUDA-bundled
installer, build the sidecar with a CUDA torch install:

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
node scripts/build-installers.mjs
```

`curl -s http://127.0.0.1:8765/health` reports the device the shipped build
picked.

## Brand assets

All logos, favicons, PWA icons, and OS icons are generated from one source
image:

```bash
python scripts/generate-brand-assets.py "/path/to/stethos logo.png"
npx tauri icon src-tauri/brand/source.png
```

---

Research tool, not a medical device — confirm findings with a clinician.
