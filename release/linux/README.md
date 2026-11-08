# linux installers — cpu

| file | best for |
| --- | --- |
| `Stethos_<version>_amd64.deb` | debian, ubuntu, mint, pop!_os, zorin |
| `Stethos-<version>-1.x86_64.rpm` | fedora, rhel, opensuse |
| `Stethos-<version>-1-x86_64.pkg.tar.zst` | arch, cachyos, manjaro, endeavouros |
| `Stethos_<version>_amd64.AppImage` | any distro, no root |

these are the **cpu** packages (~460 mb). the cuda packages for nvidia gpus (~3 gb) live in [`../cuda/linux/`](../cuda/linux/) and use the same filenames, so install one set or the other. check the recorded device in the app's status bar, or run `curl -s http://127.0.0.1:8765/health`.

## appimage (no root)

```bash
chmod +x install-linux.sh
./install-linux.sh Stethos_*.AppImage
```

this copies the appimage to `~/.local/bin/stethos.AppImage`, writes a desktop entry to `~/.local/share/applications/Stethos.desktop`, installs the hicolor icons from the appimage's bundled resources, and refreshes the desktop database. remove it again with `./install-linux.sh --uninstall`.

## verifying

```bash
sha256sum -c SHA256SUMS.txt
```