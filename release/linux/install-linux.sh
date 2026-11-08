#!/usr/bin/env bash
# Install the Stethos AppImage for the current user and register it in the
# desktop application menu. No root required.
#
#   ./install-linux.sh Stethos_0.1.0_amd64.AppImage
#   ./install-linux.sh --uninstall
set -euo pipefail

APP_NAME="Stethos"
BIN_DIR="${HOME}/.local/bin"
APP_DIR="${HOME}/.local/opt/stethos"
DESKTOP_DIR="${HOME}/.local/share/applications"
ICON_ROOT="${HOME}/.local/share/icons/hicolor"
TARGET="${BIN_DIR}/stethos.AppImage"
DESKTOP_FILE="${DESKTOP_DIR}/Stethos.desktop"

uninstall() {
  rm -f "${TARGET}" "${DESKTOP_FILE}"
  rm -rf "${APP_DIR}"
  for size in 32 64 128 256 512; do
    rm -f "${ICON_ROOT}/${size}x${size}/apps/stethos.png"
  done
  command -v update-desktop-database >/dev/null 2>&1 && \
    update-desktop-database "${DESKTOP_DIR}" >/dev/null 2>&1 || true
  echo "Stethos removed."
}

if [[ "${1:-}" == "--uninstall" ]]; then
  uninstall
  exit 0
fi

SOURCE="${1:-}"
if [[ -z "${SOURCE}" || ! -f "${SOURCE}" ]]; then
  echo "usage: $0 <Stethos-*.AppImage>" >&2
  exit 1
fi

mkdir -p "${BIN_DIR}" "${APP_DIR}" "${DESKTOP_DIR}" "${ICON_ROOT}"

echo "Installing ${SOURCE} ..."
install -m 0755 "${SOURCE}" "${TARGET}"

# Pull the icon theme out of the AppImage so the launcher has a real icon even
# when the desktop environment does not read icons from inside the image.
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
if (cd "${TMP_DIR}" && "${TARGET}" --appimage-extract >/dev/null 2>&1); then
  while IFS= read -r icon; do
    rel="${icon#${TMP_DIR}/squashfs-root/usr/share/icons/hicolor/}"
    size="${rel%%/*}"
    mkdir -p "${ICON_ROOT}/${size}/apps"
    cp "${icon}" "${ICON_ROOT}/${size}/apps/stethos.png"
  done < <(find "${TMP_DIR}/squashfs-root/usr/share/icons/hicolor" -path '*/apps/*.png' 2>/dev/null)
fi

cat > "${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Name=${APP_NAME}
Comment=Local respiratory diagnostics workstation
Exec=${TARGET}
Icon=stethos
Terminal=false
Categories=MedicalSoftware;Science;Utility;
Keywords=stethos;respiratory;chest;xray;spirometry;breath;diagnostics;
StartupNotify=true
StartupWMClass=Stethos
EOF

command -v update-desktop-database >/dev/null 2>&1 && \
  update-desktop-database "${DESKTOP_DIR}" >/dev/null 2>&1 || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && \
  gtk-update-icon-cache -f -t "${ICON_ROOT}" >/dev/null 2>&1 || true

echo "Installed. Launch \"${APP_NAME}\" from your application menu,"
echo "or run ${TARGET}"
