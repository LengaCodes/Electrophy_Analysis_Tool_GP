# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

# Manually gather all files in media/ and ui/
datas = []
for folder in ('media', 'ui'):
    for root, _, files in os.walk(folder):
        for fname in files:
            src = os.path.join(root, fname)
            # place into the same relative path in the bundle
            dest = os.path.relpath(root, folder)
            datas.append((src, os.path.join(folder, dest)))

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=['scipy', 'h5py', 'matplotlib', 'PIL', 'numpy', 'pandas', 'openpyxl'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    exclude_binaries=True,
    name='ElectrophysiologyApp',
    debug=False,
    strip=False,
    upx=True,
    console=True,
    icon='media/logoClair.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='ElectrophysiologyApp',
)
