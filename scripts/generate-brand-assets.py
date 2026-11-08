#!/usr/bin/env python3
"""Regenerate every Stethos logo/icon from one source file.

Usage:
    python scripts/generate-brand-assets.py "/path/to/stethos new logo.png"

Produces:
    public/logo.png                  full-colour mark (transparent) for light surfaces
    public/logo-light.png            light monochrome mark (transparent) for dark UI
    public/favicon.ico               16/32/48/64 favicon
    public/favicon-32.png            browser tab icon
    public/apple-touch-icon.png      180x180 touch icon
    public/icons/icon-*.png          PWA / manifest icons
    src-tauri/icons/                 Tauri desktop icons (.ico, .icns, Store logos)

The desktop/PWA icons use a white rounded plate behind the navy mark so the
logo stays legible on dark docks, taskbars and home screens.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DEFAULT = Path("/home/rajdeep/Documents/stethos new logo.png")

NAVY = (14, 47, 82)
LIGHT = (233, 240, 248)
PLATE = (255, 255, 255)

PWA_SIZES = [16, 32, 48, 64, 72, 96, 128, 144, 152, 180, 192, 256, 384, 512]


def load_mark(source: Path) -> Image.Image:
    img = Image.open(source).convert("RGBA")
    bbox = img.split()[3].getbbox()
    if not bbox:
        raise SystemExit(f"{source} has no visible pixels")
    return img.crop(bbox)


def resize(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.LANCZOS)


def monochrome(mark: Image.Image, colour: tuple[int, int, int]) -> Image.Image:
    alpha = mark.split()[3]
    solid = Image.new("RGBA", mark.size, (*colour, 255))
    solid.putalpha(alpha)
    return solid


def plate_with_mark(mark: Image.Image, size: int, *, rounded: bool, bleed: bool) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0) if rounded or not bleed else (*PLATE, 255))
    draw = ImageDraw.Draw(canvas)
    if bleed:
        draw.rectangle((0, 0, size, size), fill=(*PLATE, 255))
    else:
        margin = round(size * 0.035)
        radius = round(size * 0.22)
        draw.rounded_rectangle(
            (margin, margin, size - margin, size - margin),
            radius=radius,
            fill=(*PLATE, 255),
        )
    inner = round(size * (0.60 if bleed else 0.53))
    logo = mark.copy()
    scale = min(inner / logo.width, inner / logo.height)
    logo = logo.resize(
        (max(1, round(logo.width * scale)), max(1, round(logo.height * scale))),
        Image.LANCZOS,
    )
    x = (size - logo.width) // 2
    y = (size - logo.height) // 2
    canvas.alpha_composite(logo, (x, y))
    return canvas


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE_DEFAULT
    if not source.exists():
        raise SystemExit(f"source logo not found: {source}")

    mark = load_mark(source)
    public = ROOT / "public"
    icons = public / "icons"
    tauri_icons = ROOT / "src-tauri" / "icons"
    public.mkdir(exist_ok=True)
    icons.mkdir(exist_ok=True)
    tauri_icons.mkdir(exist_ok=True)

    # in-app marks
    for name, colour in [("logo.png", NAVY), ("logo-light.png", LIGHT)]:
        solid = monochrome(mark, colour)
        solid.thumbnail((512, 512), Image.LANCZOS)
        solid.save(public / name)
        print("wrote", public / name)

    # favicons + touch icon
    plate_with_mark(mark, 32, rounded=True, bleed=False).save(public / "favicon-32.png")
    apple = plate_with_mark(mark, 180, rounded=False, bleed=True)
    apple.save(public / "apple-touch-icon.png")
    favicon = plate_with_mark(mark, 64, rounded=True, bleed=False)
    favicon.save(
        public / "favicon.ico",
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64)],
    )
    print("wrote favicons + apple-touch-icon")

    # PWA icons: full-bleed white so maskable cropping never cuts the mark
    for size in PWA_SIZES:
        icon = plate_with_mark(mark, size, rounded=False, bleed=True)
        icon.save(icons / f"icon-{size}.png")
    print(f"wrote {len(PWA_SIZES)} PWA icons")

    # Tauri source: one square master, then let `tauri icon` derive all formats
    brand = ROOT / "src-tauri" / "brand"
    brand.mkdir(parents=True, exist_ok=True)
    master = plate_with_mark(mark, 1024, rounded=True, bleed=False)
    master.save(brand / "source.png")
    print("wrote", brand / "source.png")


if __name__ == "__main__":
    main()
