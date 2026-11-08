# macos installers — cpu

| file | notes |
| --- | --- |
| `Stethos_<version>_aarch64.dmg` | apple silicon (m1/m2/m3/m4) |
| `Stethos_<version>_aarch64.app.zip` | raw `.app` bundle |

open the `.dmg`, drag **stethos** into **applications**, and launch it from launchpad or spotlight. the bundle registers itself with launch services, so it shows up in applications immediately.

the dmg is ad-hoc signed. if gatekeeper blocks the first launch, open **system settings → privacy & security** and choose **open anyway** once.

the bundled inference engine uses metal (mps) on apple silicon and falls back to the cpu on intel.

```bash
shasum -a 256 -c SHA256SUMS.txt
```