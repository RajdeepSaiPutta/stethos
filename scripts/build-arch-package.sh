#!/usr/bin/env bash
# Build an Arch Linux package (.pkg.tar.zst) from the artifacts Tauri already
# produced. Arch is not a Tauri bundle target, but a pacman package is the only
# way to get a proper application-menu entry on Arch-based distros.
#
#   node scripts/build-installers.mjs --bundles deb
#   ./scripts/build-arch-package.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(node -p "require('${ROOT}/package.json').version")"
RELEASE=1
ARCH="$(uname -m)"
STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

BIN="${ROOT}/src-tauri/target/release/stethos"
SIDECAR="${SIDECAR:-${ROOT}/python/dist-cpu/stethos-sidecar}"
ICONS="${ROOT}/src-tauri/icons"
OUT_DIR="${OUT_DIR:-${ROOT}/release/linux}"

for f in "${BIN}" "${SIDECAR}"; do
  [[ -f "${f}" ]] || { echo "missing ${f} - build the app first" >&2; exit 1; }
done

mkdir -p "${STAGE}/usr/bin" "${STAGE}/usr/share/applications"
install -m 0755 "${BIN}" "${STAGE}/usr/bin/stethos"
install -m 0755 "${SIDECAR}" "${STAGE}/usr/bin/stethos-sidecar"

for size in 32 64 128 256 512; do
  src="${ICONS}/${size}x${size}.png"
  [[ -f "${src}" ]] || continue
  mkdir -p "${STAGE}/usr/share/icons/hicolor/${size}x${size}/apps"
  install -m 0644 "${src}" "${STAGE}/usr/share/icons/hicolor/${size}x${size}/apps/stethos.png"
done

cat > "${STAGE}/usr/share/applications/Stethos.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=Stethos
Comment=Local respiratory diagnostics workstation
Exec=stethos
Icon=stethos
Terminal=false
Categories=MedicalSoftware;Science;Utility;
Keywords=stethos;respiratory;chest;xray;spirometry;breath;diagnostics;
StartupNotify=true
StartupWMClass=Stethos
EOF

INSTALLED_SIZE="$(du -sb "${STAGE}" | cut -f1)"
BUILD_DATE="$(date +%s)"

cat > "${STAGE}/.PKGINFO" <<EOF
pkgname = stethos
pkgbase = stethos
pkgver = ${VERSION}-${RELEASE}
pkgdesc = Local respiratory diagnostics workstation
url = https://github.com/RajdeepSaiPutta/stethos
builddate = ${BUILD_DATE}
packager = Stethos
size = ${INSTALLED_SIZE}
arch = ${ARCH/x86_64/x86_64}
license = MIT
depend = webkit2gtk-4.1
depend = gtk3
EOF

# .MTREE is what pacman uses for file integrity; generate it with bsdtar when
# available (it is part of libarchive).
if command -v bsdtar >/dev/null 2>&1; then
  ( cd "${STAGE}" && bsdtar -czf .MTREE --format=mtree \
      --options='!all,use-set,type,uid,gid,mode,time,size,md5,sha256,link' \
      .PKGINFO usr )
fi

mkdir -p "${OUT_DIR}"
OUT="${OUT_DIR}/Stethos-${VERSION}-${RELEASE}-${ARCH}.pkg.tar.zst"

if command -v zstd >/dev/null 2>&1; then
  ( cd "${STAGE}" && tar -cf - .PKGINFO .MTREE usr 2>/dev/null || tar -cf - .PKGINFO usr ) \
    | zstd -19 -T0 -q -o "${OUT}"
else
  ( cd "${STAGE}" && tar --zstd -cf "${OUT}" .PKGINFO .MTREE usr )
fi

echo "Built ${OUT}"
echo "Install with: sudo pacman -U '${OUT}'"
