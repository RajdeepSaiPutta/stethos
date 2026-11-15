#!/usr/bin/env bash
# Stethos — one-line install
#   curl -fsSL https://raw.githubusercontent.com/RajdeepSaiPutta/stethos/main/install.sh | bash
# or: wget -qO- https://raw.githubusercontent.com/RajdeepSaiPutta/stethos/main/install.sh | bash

set -euo pipefail

REPO="rajdeepp11/stethos"
HF_BASE="https://huggingface.co/$REPO/resolve/main"
INSTALL_DIR="$HOME/.local/bin"
DESKTOP_DIR="$HOME/.local/share/applications"
ICON_DIR="$HOME/.local/share/icons/hicolor/64x64/apps"

echo "Installing Stethos..."

mkdir -p "$INSTALL_DIR" "$DESKTOP_DIR" "$ICON_DIR"

# Detect distro for package preference
if command -v dpkg >/dev/null 2>&1; then
    PKG="deb"
    PKG_NAME="Stethos_0.1.0_amd64.deb"
elif command -v rpm >/dev/null 2>&1; then
    PKG="rpm"
    PKG_NAME="Stethos-0.1.0-1.x86_64.rpm"
elif command -v pacman >/dev/null 2>&1; then
    PKG="arch"
    PKG_NAME="Stethos-0.1.0-1-x86_64.pkg.tar.zst"
else
    PKG="appimage"
    PKG_NAME="Stethos_0.1.0_amd64.AppImage"
fi

echo "Downloading $PKG package..."
cd /tmp
if command -v wget >/dev/null 2>&1; then
    wget -q "$HF_BASE/$PKG_NAME" -O "$PKG_NAME"
else
    curl -fsSL "$HF_BASE/$PKG_NAME" -o "$PKG_NAME"
fi

echo "Installing..."
case $PKG in
    deb)   sudo dpkg -i "$PKG_NAME" || sudo apt-get install -f -y ;;
    rpm)   sudo rpm -i "$PKG_NAME" ;;
    arch)  sudo pacman -U --noconfirm "$PKG_NAME" ;;
    appimage)
        cp "$PKG_NAME" "$INSTALL_DIR/stethos"
        chmod +x "$INSTALL_DIR/stethos"
        # Extract icon & desktop file from AppImage
        "$INSTALL_DIR/stethos" --appimage-extract >/dev/null 2>&1
        cp squashfs-root/stethos.png "$ICON_DIR/stethos.png" 2>/dev/null || true
        cat > "$DESKTOP_DIR/stethos.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Stethos
Comment=Local respiratory diagnostics workstation
Exec=$INSTALL_DIR/stethos
Icon=stethos
Terminal=false
Categories=MedicalSoftware;Science;Utility;
StartupNotify=true
StartupWMClass=Stethos
DESK
        update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true
        rm -rf squashfs-root
        ;;
esac

echo "Done! Launch from app menu or run: stethos"
