# linux installers — cuda build

these packages bundle the cuda 13 inference engine, so the app runs every model on an nvidia gpu (`"device": "cuda"` on `/health`). they are ~2.9–3.1 gb each, because the cuda runtime and kernels are inside the package.

| file | best for |
| --- | --- |
| `Stethos_<version>_amd64.deb` | debian, ubuntu, mint, pop!_os, zorin |
| `Stethos-<version>-1.x86_64.rpm` | fedora, rhel, opensuse |
| `Stethos-<version>-1-x86_64.pkg.tar.zst` | arch, cachyos, manjaro, endeavouros |
| `Stethos_<version>_amd64.AppImage` | any distro, no root |

the filenames match the cpu build in `release/linux/`. install **one or the other**, not both. if in doubt, the cpu build is ~460 mb and runs on any machine; use the cuda build only when you have a supported nvidia gpu.

## requirements

- an nvidia gpu supported by cuda 13 (geforce rtx 20-series or newer works).
- a driver that supports cuda 13.0 — **580 or newer**. check with `nvidia-smi`; the app falls back to cpu when the driver or gpu is missing, so it still launches.

## what gets installed

every package installs `/usr/bin/stethos`, the cuda inference engine next to it, the icon set under `/usr/share/icons/hicolor/`, and `/usr/share/applications/Stethos.desktop`, so stethos appears in the desktop application menu with its logo.

## verify

```bash
sha256sum -c SHA256SUMS.txt
```

check the device the installed build picked:

```bash
curl -s http://127.0.0.1:8765/health
```

`"device": "cuda"` and a non-null `gpu_name` mean the gpu path is active.

## uninstall

```bash
sudo apt remove stethos        # deb
sudo dnf remove stethos        # rpm
sudo pacman -R stethos         # arch
```

for the appimage, run `../linux/install-linux.sh --uninstall`.