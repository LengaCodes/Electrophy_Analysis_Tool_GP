import sys
from cx_Freeze import setup, Executable

build_exe_options = {
    "packages": ["os", "tkinter", "ui", "numpy", "pandas", "matplotlib", "h5py", "scipy", "PIL"],
    "include_files": ["media/logoClair.ico", "media/LogoVclair.png"],
}

exe = Executable(
    script="main.py",
    base="Win32GUI",  # no console
    icon="media/logoClair.ico",
)

setup(
    name="ElectrophysiologyApp",
    version="1.0",
    description="Electrophysiology Signal Analysis",
    options={"build_exe": build_exe_options},
    executables=[exe],
)
