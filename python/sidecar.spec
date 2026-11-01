# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Core dependencies that must be included
hiddenimports = [
    'fastapi',
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'uvicorn.lifespan.off',
    'pydantic',
    'pydantic.v1',
    'pydantic_core',
    'pydantic_settings',
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torchvision',
    'torchvision.models',
    'torchaudio',
    'numpy',
    'scipy',
    'sklearn',
    'sklearn.preprocessing',
    'sklearn.metrics',
    'albumentations',
    'cv2',
    'PIL',
    'pydicom',
    'SimpleITK',
    'monai',
    'fhir.resources',
    'reportlab',
    'jinja2',
    'yaml',
    'dotenv',
    'requests',
    'tqdm',
    'librosa',
    'soundfile',
    'pandas',
    'packaging',
]

# Data files to include (model weights) - from the repo root
datas = [
    ('./models', 'models'),  # python/models
    # Include the plugins package
    ('./plugins', 'plugins'),
    # Include inference dir
    ('./inference', 'inference'),
]

# Include NVIDIA CUDA libraries from the local torch install
import os
nvidia_lib_path = os.path.expanduser('~/.local/lib/python3.13/site-packages/nvidia')
torch_lib_path = os.path.expanduser('~/.local/lib/python3.13/site-packages/torch/lib')
binaries = []
for lib_path, prefix in [(nvidia_lib_path, 'nvidia'), (torch_lib_path, 'torch/lib')]:
    if os.path.exists(lib_path):
        for root, dirs, files in os.walk(lib_path):
            for f in files:
                if f.endswith('.so') or f.endswith('.so.') or '.so.' in f:
                    src = os.path.join(root, f)
                    rel = os.path.relpath(root, lib_path)
                    dst = os.path.join(prefix, rel) if rel != '.' else prefix
                    binaries.append((src, dst))

a = Analysis(
    ['api/main.py'],
    pathex=[os.path.abspath('..')],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'notebook',
        'jupyter',
        'IPython',
        'pytest',
        'sphinx',
        'docutils',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out unnecessary torch/cuda binaries to reduce size
# Keep only what's needed for inference
a.binaries = [x for x in a.binaries if not any(skip in x[0] for skip in [
    'libiomp', 'libgomp',
])]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='stethos-sidecar',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)