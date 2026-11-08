# stethos

stethos is a desktop app for respiratory diagnostics. it takes breath sounds, a chest x-ray, or spirometry data, classifies each one, and fuses the findings into a single risk assessment with a per-modality breakdown. reports export as pdf, fhir, dicom-sr, or json.

all inference runs locally in a python sidecar. it uses cuda when an nvidia gpu is present and falls back to cpu otherwise; nothing ever leaves the machine. where no trained checkpoint exists yet, the models fall back to locally-initialized weights so the whole pipeline stays runnable end to end.

built with tauri 2, react, and fastapi.

## install

download the installer for your platform:

- **github releases (cpu, all platforms):**
  <https://github.com/RajdeepSaiPutta/stethos/releases/tag/v0.1.0>
- **hugging face (cpu + cuda linux):**
  <https://huggingface.co/rajdeepp11/stethos>

| platform | installer |
| --- | --- |
| windows | `Stethos_<version>_x64-setup.exe` |
| macos | `Stethos_<version>_aarch64.dmg` |
| debian / ubuntu | `Stethos_<version>_amd64.deb` |
| fedora / opensuse | `Stethos-<version>-1.x86_64.rpm` |
| arch / cachyos | `Stethos-<version>-1-x86_64.pkg.tar.zst` |
| any linux | `install-linux.sh` + AppImage |
| linux + nvidia gpu | cuda set under `cuda/linux/` (hugging face only) |

each installer registers stethos in the application menu / start menu / launchpad and bundles the inference engine plus the model weights. no terminal, no python, no separate model download. see [`release/README.md`](release/README.md) for checksums, gpu notes, and uninstall instructions.

## run

**installed app:** launch from your app drawer / start menu / applications. the status bar (bottom right) shows `sidecar connected · models 4/4` and either `cpu` or your gpu name + memory.

**verify gpu path:**
```bash
curl -s http://127.0.0.1:8765/health
```
`"device": "cuda"` with a `gpu_name` means gpu acceleration is active. `"device": "cpu"` means it's running on cpu (or metal/mps on apple silicon).

## build from source

```bash
npm ci
node scripts/build-installers.mjs                # installer for this os
```

or run in development (two terminals):

```bash
# terminal 1 - inference sidecar
cd python && python -m uvicorn api.main:app --port 8765

# terminal 2 - desktop shell
npm run tauri:dev
```

### gpu acceleration (local build only)

the packaged sidecar detects cuda at runtime. a cpu-only pytorch wheel still runs everything, but reports `"device": "cpu"`. to produce a cuda-bundled installer, build the sidecar with a cuda torch install:

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
node scripts/build-installers.mjs
```

note: cuda builds are ~3 gb and only work on linux (built locally). ci builds are cpu-only because the one-file pyinstaller/nsis pipeline can't handle binaries >2 gb, and ci runners have no gpus to test.

## brand assets

all logos, favicons, pwa icons, and os icons are generated from one source image:

```bash
python scripts/generate-brand-assets.py "/path/to/stethos logo.png"
npx tauri icon src-tauri/brand/source.png
```

---

research tool, not a medical device — confirm findings with a clinician.