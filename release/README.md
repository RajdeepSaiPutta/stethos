# stethos installers

ready-to-run installers for **linux**, **windows** and **macos**. each one is a normal desktop install: double-click it, follow the wizard, and stethos appears in the app drawer / start menu / applications folder. no terminal, no python, no separate model download — the inference engine and model weights are inside the package.

## download

- **github releases (cpu, all platforms):**
  <https://github.com/RajdeepSaiPutta/stethos/releases/tag/v0.1.0>
- **hugging face (cpu + cuda linux):**
  <https://huggingface.co/rajdeepp11/stethos>

the binaries are also in the folders below on a machine that built them.

```
release/
├── linux/            cpu  .deb  .rpm  .AppImage  .pkg.tar.zst  (+ install-linux.sh)
├── windows/          cpu  .exe (nsis)  .msi
├── macos/            cpu  .dmg
└── cuda/
    └── linux/        cuda .deb  .rpm  .AppImage  .pkg.tar.zst
```

every platform folder gets a `SHA256SUMS.txt` next to the installers. verify with:

```bash
sha256sum -c SHA256SUMS.txt        # linux
shasum -a 256 -c SHA256SUMS.txt    # macos
certutil -hashfile <file> SHA256   # windows
```

## which build?

| build | platforms | size | notes |
| --- | --- | --- | --- |
| **cpu** | linux, windows, macos | ~330–500 mb | runs everywhere. macos uses metal (mps) on apple silicon automatically. |
| **cuda** | linux x64 | ~2.9–3.1 gb | runs every model on an nvidia gpu. requires a cuda 13 capable driver (580 or newer). |

the cpu and cuda linux packages have the same filenames, so install **one or the other**, not both. if you don't have an nvidia gpu, use the cpu build.

## install

### linux

| distro family | file | command |
| --- | --- | --- |
| debian / ubuntu / mint | `Stethos_<version>_amd64.deb` | double-click, or `sudo apt install ./Stethos_*.deb` |
| fedora / opensuse | `Stethos-<version>-1.x86_64.rpm` | double-click, or `sudo dnf install ./Stethos-*.rpm` |
| arch / cachyos / manjaro | `Stethos-<version>-1-x86_64.pkg.tar.zst` | `sudo pacman -U Stethos-*.pkg.tar.zst` |
| any distro, no root | `Stethos_<version>_amd64.AppImage` | `./release/linux/install-linux.sh Stethos_*.AppImage` |

all four register a desktop entry (`MedicalSoftware;Science;`) so the app shows up in the application menu and can be pinned to the dock. the appimage helper installs it per-user under `~/.local` — no administrator password needed.

for the cuda packages, use the same commands from `release/cuda/linux/`.

### windows

run `Stethos_<version>_x64-setup.exe` (recommended) or `Stethos_<version>_x64_en-US.msi`. the installer creates a start-menu shortcut and an optional desktop shortcut.

### macos

open `Stethos_<version>_aarch64.dmg` (apple silicon), drag stethos into **applications**, then launch it from launchpad. the build is ad-hoc signed; on first launch use **system settings → privacy & security → open anyway** if gatekeeper asks.

## gpu acceleration

stethos uses the gpu automatically when one is available and falls back to cpu otherwise. the installer bundles the inference engine that matches its build:

- **cuda build** (linux x64): ships the cuda runtime and nvidia kernels, so `/health` reports `"device": "cuda"`. these packages are ~2.9–3.1 gb.
- **cpu build** (linux, windows, macos): ~330–500 mb. on apple silicon pytorch enables metal (mps) automatically; elsewhere it runs on cpu.

to check what an installed build picked, open the app and look at the status bar (bottom right: `gpu name · memory` or `cpu`), or run:

```bash
curl -s http://127.0.0.1:8765/health
```

## rebuilding

from a clean checkout:

```bash
npm ci
node scripts/build-installers.mjs                  # cpu build, host platform
node scripts/build-installers.mjs --no-sidecar     # shell-only, fast
PYTHON=/path/to/cuda/python node scripts/build-installers.mjs   # cuda build (local only)
```

the script builds the web bundle, runs pyinstaller for the sidecar (unless it is already in `python/dist/`), runs `tauri build` for the host platform, and copies every artifact plus `SHA256SUMS.txt` into `release/<platform>/`.

the arch package is built from the same artifacts:

```bash
./scripts/build-arch-package.sh
```

### why the binaries are not committed to git

github rejects tracked files over 100 mb and the cuda installers are ~3 gb, so installer binaries are excluded from the repository (see `.gitignore`). the cpu set is produced by `.github/workflows/build-installers.yml` and attached to the github **releases** page, which allows assets up to 2 gb each. the full set (cpu + cuda linux) is mirrored to hugging face, which allows larger files. the local `release/` folders on a build machine still contain the real files.

## platform notes

- **linux runtime**: `libwebkit2gtk-4.1`, `libgtk-3-0` (declared as deb dependencies; the rpm and arch packages use the equivalent names).
- **macos**: 11.0 or newer, apple silicon and intel both supported.
- **windows**: windows 10/11 x64. the nsis installer installs per-user, so it does not need an administrator account.