# stethos

local multimodal respiratory ml workstation. desktop app (tauri + rust) with bundled python sidecar for inference.

## quick install

```bash
# one-liner (linux)
curl -fsSL https://raw.githubusercontent.com/RajdeepSaiPutta/stethos/main/install.sh | bash

# or download from GitHub Releases
# https://github.com/RajdeepSaiPutta/stethos/releases
```

## what's inside

| component | description |
|-----------|-------------|
| **breath sounds** | cnn classifier (normal / crackle / wheeze / both) |
| **chest x-ray** | densenet121 classifier (14 pathologies) + mobilenetv3 u-net segmenter (lungs, heart, trachea) |
| **spirometry** | pattern classifier (normal / obstructive / restrictive / mixed) |
| **fusion** | combines all three modalities |
| **3d lung viewer** | volumetric rendering + mpr quad view + segmentation-driven lung meshes |

## download (cpu installers)

| package | size | distro |
|---------|------|--------|
| `Stethos_0.1.0_amd64.deb` | 960 MB | debian/ubuntu/mint |
| `Stethos-0.1.0-1.x86_64.rpm` | 960 MB | fedora/rhel/opensuse |
| `Stethos_0.1.0_amd64.AppImage` | 566 MB | any linux (portable) |
| `Stethos-0.1.0-1-x86_64.pkg.tar.zst` | 470 MB | arch/manjaro |
| `Stethos_0.1.0_x64.msi` | ~1 GB | windows 10/11 |
| `Stethos_0.1.0_x64-setup.exe` | ~1 GB | windows 10/11 (NSIS) |
| `Stethos_0.1.0_aarch64.dmg` | ~1 GB | macOS 11+ (Apple Silicon) |

all on GitHub Releases: **https://github.com/RajdeepSaiPutta/stethos/releases**

cuda sidecar (3.0 GB) not packaged — run `python/dist-cuda/stethos-sidecar` standalone + dev binary.

## test data

realistic synthetic datasets (included in `test-data/`):

- `breath/` — 39 wav files, 13 conditions
- `cxr/` — 32 png images, 16 pathologies  
- `spirometry/` — 32 patients × 4 visits, csv + json

## 3d viewer features

- **quad mpr**: axial / sagittal / coronal / 3d volume
- **volume rendering**: 72 additive slices, window/level presets (lung, mediastinum, bone, soft tissue)
- **lung meshes**: built from segmentation mask (distance transform → bulge heightfield → alpha-mapped surface)
- **render modes**: surface, wireframe, points, volume, **lungs** (anatomical lobes)
- **measurements**: distance (mm), angle (°), roi
- **colormaps**: radiograph, bone, thermal, hot-iron, mammography, inverted

## build from source

```bash
# prerequisites: rust, node, python 3.11+, cuda 12 (optional)
git clone https://github.com/RajdeepSaiPutta/stethos
cd stethos

# cpu sidecar
cd python && pip install -r requirements-sidecar.txt --extra-index-url https://download.pytorch.org/whl/cpu
pyinstaller --clean -y sidecar.spec
cp dist/stethos-sidecar ../src-tauri/binaries/

# app
npm ci && npm run build
cd src-tauri && cargo build --release
# or: cargo tauri build --bundles deb,appimage,rpm
```

## project structure

```
├── src/                    # tauri frontend (react + typescript + three.js)
│   ├── components/
│   │   ├── Xray3DViewer.tsx    # 3d mpr + volume + lung meshes
│   │   ├── ModalityPanel.tsx   # per-modality results + 3d toggle
│   │   └── ...
├── src-tauri/              # rust backend + tauri config
│   ├── src/
│   │   ├── main.rs           # commands, sidecar management
│   │   └── sidecar_manager.rs
│   └── tauri.conf.json
├── python/                 # fastapi sidecar
│   ├── api/main.py         # endpoints: /health, /inference, /fusion, /device
│   ├── plugins/
│   │   ├── cxr.py          # classification + segmentation (returns mask)
│   │   ├── breath.py
│   │   ├── spirometry.py
│   │   └── fusion.py
│   ├── make_realistic_*.py # test data generators
│   └── dist-cpu/           # cpu sidecar binary
├── test-data/              # sample inputs (git)
├── install.sh              # universal one-line installer
└── scripts/                # installer builders
```

## citation

```
@software{stethos,
  author = {Rajdeep Sai Putta},
  title = {stethos: local multimodal respiratory ml workstation},
  year = {2026},
  url = {https://github.com/RajdeepSaiPutta/stethos}
}
```
