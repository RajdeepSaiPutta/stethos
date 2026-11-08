# Linux installers — CUDA build

These packages bundle the CUDA 13 inference engine, so the app runs every model
on an NVIDIA GPU (`"device": "cuda"` on `/health`). They are ~2.9–3.1 GB each,
because the CUDA runtime and kernels are inside the package.

| File | Best for |
| --- | --- |
| `Stethos_<version>_amd64.deb` | Debian, Ubuntu, Mint, Pop!_OS, Zorin |
| `Stethos-<version>-1.x86_64.rpm` | Fedora, RHEL, openSUSE |
| `Stethos-<version>-1-x86_64.pkg.tar.zst` | Arch, CachyOS, Manjaro, EndeavourOS |
| `Stethos_<version>_amd64.AppImage` | Any distro, no root |

The filenames match the CPU build in `release/linux/`. Install **one or the
other**, not both. If in doubt, the CPU build is ~460 MB and runs on any
machine; use the CUDA build only when you have a supported NVIDIA GPU.

## Requirements

- An NVIDIA GPU supported by CUDA 13 (GeForce RTX 20-series or newer works).
- A driver that supports CUDA 13.0 — **580 or newer**. Check with
  `nvidia-smi`; the app falls back to CPU when the driver or GPU is missing, so
  it still launches.

## What gets installed

Every package installs `/usr/bin/stethos`, the CUDA inference engine next to it,
the icon set under `/usr/share/icons/hicolor/`, and
`/usr/share/applications/Stethos.desktop`, so Stethos appears in the desktop
application menu with its logo.

## Verify

```bash
sha256sum -c SHA256SUMS.txt
```

Check the device the installed build picked:

```bash
curl -s http://127.0.0.1:8765/health
```

`"device": "cuda"` and a non-null `gpu_name` mean the GPU path is active.

## Uninstall

```bash
sudo apt remove stethos        # deb
sudo dnf remove stethos        # rpm
sudo pacman -R stethos         # Arch
```

For the AppImage, run `../linux/install-linux.sh --uninstall`.
