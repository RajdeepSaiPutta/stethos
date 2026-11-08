# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the bundled Python sidecar.

Build (from the ``python/`` directory):

    pyinstaller --clean -y sidecar.spec

The output is a single self-contained executable at
``python/dist/stethos-sidecar`` (``.exe`` on Windows).  Install a *CPU* torch
wheel before building to keep the bundle small::

    pip install -r requirements-sidecar.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

CUDA is auto-detected at runtime; a CPU-only build simply reports ``cpu``.
"""

import os
import sys
from pathlib import Path

block_cipher = None

HERE = Path(SPECPATH).resolve()

hiddenimports = [
    "fastapi",
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "pydantic",
    "pydantic.v1",
    "pydantic_core",
    "pydantic_settings",
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torchvision",
    "torchvision.extension",
    "torchvision.ops",
    "torchvision.models",
    "torchaudio",
    "numpy",
    "scipy",
    "sklearn",
    "sklearn.preprocessing",
    "sklearn.metrics",
    "cv2",
    "PIL",
    "pydicom",
    "fhir.resources",
    "reportlab",
    "jinja2",
    "yaml",
    "dotenv",
    "pandas",
    "packaging",
    "python_multipart",
]

datas = [
    (str(HERE / "models"), "models"),
    (str(HERE / "plugins"), "plugins"),
]

binaries = []

# torchvision >= 0.23 loads its C++ operator library at runtime through
# ``torch.ops.load_library`` rather than a normal Python extension import, so
# PyInstaller cannot discover it. Ship the shared objects explicitly.
try:
    import torchvision

    tv_dir = Path(torchvision.__file__).resolve().parent
    for pattern in ("_C*.so", "_C*.pyd", "image*.so", "image*.pyd"):
        for path in tv_dir.glob(pattern):
            binaries.append((str(path), "torchvision"))
except Exception:  # pragma: no cover - build-time convenience only
    pass

# Ship the NVIDIA userspace libraries only when torch is a CUDA build. They
# live in the same site-packages as torch (a venv, or the user site).  A CPU
# wheel has no such directory, which keeps that installer ~1 GB smaller.
try:
    import torch

    torch_is_cuda = bool(getattr(getattr(torch, "version", None), "cuda", None))

    if torch_is_cuda:
        torch_site = Path(torch.__file__).resolve().parent.parent
        nvidia = torch_site / "nvidia"
        if nvidia.is_dir():
            for root, _dirs, files in os.walk(nvidia):
                for f in files:
                    if ".so" not in f and ".dll" not in f:
                        continue
                    src = Path(root) / f
                    rel = src.relative_to(nvidia)
                    binaries.append((str(src), str(Path("nvidia") / rel.parent)))
except Exception:  # pragma: no cover - build-time convenience only
    pass

a = Analysis(
    [str(HERE / "api" / "main.py")],
    pathex=[str(HERE)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "notebook",
        "jupyter",
        "IPython",
        "pytest",
        "sphinx",
        "docutils",
        "tests",
        "SimpleITK",
        "monai",
        "albumentations",
        "librosa",
        "soundfile",
        "numba",
        "llvmlite",
        "torch.utils.tensorboard",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="stethos-sidecar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
