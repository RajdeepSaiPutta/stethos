# Stethos installers

Ready-to-run installers for **Linux**, **Windows** and **macOS**. Each one is a
normal desktop install: double-click it, follow the wizard, and Stethos appears
in the app drawer / Start menu / Applications folder. No terminal, no Python,
no separate model download — the inference engine and model weights are inside
the package.

```
release/
├── linux/      .deb  .rpm  .AppImage  .pkg.tar.zst  (+ install-linux.sh)
├── windows/    .exe (NSIS)  .msi
└── macos/      .dmg  .app.zip
```

Every folder gets a `SHA256SUMS.txt` next to the installers. Verify with:

```bash
sha256sum -c SHA256SUMS.txt        # Linux
shasum -a 256 -c SHA256SUMS.txt    # macOS
certutil -hashfile <file> SHA256   # Windows
```

## Install

### Linux

| Distro family | File | Command |
| --- | --- | --- |
| Debian / Ubuntu / Mint | `Stethos_<version>_amd64.deb` | double-click, or `sudo apt install ./Stethos_*.deb` |
| Fedora / openSUSE | `Stethos-<version>-1.x86_64.rpm` | double-click, or `sudo dnf install ./Stethos-*.rpm` |
| Arch / CachyOS / Manjaro | `Stethos-<version>-1-x86_64.pkg.tar.zst` | `sudo pacman -U Stethos-*.pkg.tar.zst` |
| Any distro, no root | `Stethos_<version>_amd64.AppImage` | `./release/linux/install-linux.sh Stethos_*.AppImage` |

All four register a desktop entry (`MedicalSoftware;Science;`) so the app shows
up in the application menu and can be pinned to the dock. The AppImage helper
installs it per-user under `~/.local` — no administrator password needed.

### Windows

Run `Stethos_<version>_x64-setup.exe` (recommended) or
`Stethos_<version>_x64_en-US.msi`. The installer creates a Start-menu shortcut
and an optional desktop shortcut.

### macOS

Open `Stethos_<version>_aarch64.dmg` (Apple silicon), drag Stethos into
**Applications**, then launch it from Launchpad. The build is ad-hoc signed; on
first launch use **System Settings → Privacy & Security → Open Anyway** if
Gatekeeper asks.

## GPU acceleration

Stethos uses the GPU automatically when one is available and falls back to CPU
otherwise. The installer bundles the inference engine that matches its build:

- **CUDA build** (Linux x64, Windows x64): ships the CUDA runtime and NVIDIA
  kernels, so `/health` reports `"device": "cuda"`. These packages are ~2.5 GB.
- **CPU build** (default CI build, macOS): ~460 MB. On Apple silicon PyTorch
  enables Metal (MPS) automatically; elsewhere it runs on CPU.

To check what an installed build picked, open the app and look at the status
bar (bottom right: `GPU name · memory` or `cpu`), or run:

```bash
curl -s http://127.0.0.1:8765/health
```

## Rebuilding

From a clean checkout:

```bash
npm ci
node scripts/build-installers.mjs                  # CPU build, host platform
node scripts/build-installers.mjs --no-sidecar     # shell-only, fast
PYTHON=/path/to/cuda/python node scripts/build-installers.mjs   # CUDA build
```

The script builds the web bundle, runs PyInstaller for the sidecar (unless it is
already in `python/dist/`), runs `tauri build` for the host platform, and copies
every artifact plus `SHA256SUMS.txt` into `release/<platform>/`.

### Why the binaries are not committed to git

GitHub rejects tracked files over 100 MB and the CUDA installers are ~2.5 GB, so
installer binaries are excluded from the repository (see `.gitignore`). They
are produced by `.github/workflows/build-installers.yml` and attached to the
GitHub **Releases** page, which allows assets up to 2 GB each. The local
`release/` folders on a build machine still contain the real files.

## Platform notes

- **Linux runtime**: `libwebkit2gtk-4.1`, `libgtk-3-0` (declared as deb
  dependencies; the RPM and Arch packages use the equivalent names).
- **macOS**: 11.0 or newer, Apple silicon and Intel both supported.
- **Windows**: Windows 10/11 x64. The NSIS installer installs per-user, so it
  does not need an administrator account.
