# Windows installers

| File | Notes |
| --- | --- |
| `Stethos_<version>_x64-setup.exe` | NSIS installer, per-user, recommended |
| `Stethos_<version>_x64_en-US.msi` | MSI for managed / group-policy installs |

Double-click the installer and follow the wizard. It creates a Start-menu entry
(with the Stethos logo) and an optional desktop shortcut, so the app is always
launchable without a terminal.

The installer bundles the Python inference engine and the model weights. A
CUDA-enabled build reports `"device": "cuda"` on `http://127.0.0.1:8765/health`
and uses the GPU automatically; otherwise inference runs on the CPU.

Verify the download before installing:

```powershell
Get-FileHash .\Stethos_0.1.0_x64-setup.exe -Algorithm SHA256
```
