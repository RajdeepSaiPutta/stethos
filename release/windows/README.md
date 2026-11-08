# windows installers — cpu

| file | notes |
| --- | --- |
| `Stethos_<version>_x64-setup.exe` | nsis installer, per-user, recommended |
| `Stethos_<version>_x64_en-US.msi` | msi for managed / group-policy installs |

double-click the installer and follow the wizard. it creates a start-menu entry (with the stethos logo) and an optional desktop shortcut, so the app is always launchable without a terminal.

the installer bundles the python inference engine and the model weights. a cuda-enabled build would report `"device": "cuda"` on `http://127.0.0.1:8765/health` and use the gpu automatically; otherwise inference runs on the cpu. (cuda windows builds don't exist yet — the one-file pyinstaller/nsis pipeline can't handle 3 gb binaries.)

verify the download before installing:

```powershell
Get-FileHash .\Stethos_0.1.0_x64-setup.exe -Algorithm SHA256
```