# macOS installers

| File | Notes |
| --- | --- |
| `Stethos_<version>_aarch64.dmg` | Apple silicon (M1/M2/M3/M4) |
| `Stethos_<version>_aarch64.app.zip` | Raw `.app` bundle |

The default CI build targets Apple silicon. An Intel (`x64`) DMG uses the same
workflow on an Intel runner.

Open the `.dmg`, drag **Stethos** into **Applications**, and launch it from
Launchpad or Spotlight. The bundle registers itself with Launch Services, so it
shows up in Applications immediately.

The DMG is ad-hoc signed. If Gatekeeper blocks the first launch, open
**System Settings → Privacy & Security** and choose **Open Anyway** once.

The bundled inference engine uses Metal (MPS) on Apple silicon and falls back to
the CPU on Intel.

```bash
shasum -a 256 -c SHA256SUMS.txt
```
