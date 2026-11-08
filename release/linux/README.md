# Linux installers

| File | Best for |
| --- | --- |
| `Stethos_<version>_amd64.deb` | Debian, Ubuntu, Mint, Pop!_OS, Zorin |
| `Stethos-<version>-1.x86_64.rpm` | Fedora, RHEL, openSUSE |
| `Stethos-<version>-1-x86_64.pkg.tar.zst` | Arch, CachyOS, Manjaro, EndeavourOS |
| `Stethos_<version>_amd64.AppImage` | Any distro, no root |

These are the **CPU** packages (~460 MB). The CUDA packages for NVIDIA GPUs
(~3 GB) live in [`../cuda/linux/`](../cuda/linux/) and use the same filenames, so
install one set or the other. Check the recorded device in the app's status
bar, or run `curl -s http://127.0.0.1:8765/health`.

Every package installs `/usr/bin/stethos`, the inference engine next to it, the
icon set under `/usr/share/icons/hicolor/`, and
`/usr/share/applications/Stethos.desktop`. The app then appears in the desktop
application menu with the Stethos logo.

## AppImage (no root)

```bash
chmod +x install-linux.sh
./install-linux.sh Stethos_0.1.0_amd64.AppImage
```

This copies the AppImage to `~/.local/bin/stethos.AppImage`, writes a desktop
entry to `~/.local/share/applications/Stethos.desktop`, installs the hicolor
icons from the AppImage's bundled resources, and refreshes the desktop database.
Remove it again with `./install-linux.sh --uninstall`.

## Verifying

```bash
sha256sum -c SHA256SUMS.txt
```
