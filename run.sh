#!/usr/bin/env bash
#
# Startup script for the Droplet Relaxer: creates a virtual environment
# if needed, installs the dependencies (only once), and starts main.py
# -- all in a single command, without you having to activate a venv
# yourself.
#
# Usage:
#   ./run.sh
#
# If the script is not executable yet:
#   chmod +x run.sh
#   ./run.sh
#
# Troubleshooting (Linux, OpenGL/VTK window errors such as
# "Could not find a decent config" or a crash on startup): see the
# "Troubleshooting" section in README.md.

set -euo pipefail

# Directory this script lives in (so run.sh can be called from
# anywhere, not just from inside this folder).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="venv"
MARKER_FILE="$VENV_DIR/.requirements_installed"

# Find a Python interpreter (python3 preferred, else python).
# If a Conda environment is active, prefer the SYSTEM Python for the venv:
# a venv built on Conda's Python loads Conda's bundled libraries (libstdc++,
# libGL/X11 ...) and VTK then fails with "Could not find a decent config"
# and a segmentation fault (confirmed on a real laptop: rebuilding the venv
# with the system Python fixed it). Override with DROPLET_PYTHON=/path/to/python3.
if [ -n "${DROPLET_PYTHON:-}" ]; then
    PYTHON_BIN="$DROPLET_PYTHON"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x /usr/bin/python3 ]; then
    PYTHON_BIN="/usr/bin/python3"
    echo "Conda environment active: using the system Python /usr/bin/python3 for the venv."
    if [ -d "venv" ] && [ -f "venv/pyvenv.cfg" ] && grep -qi "conda\|miniconda\|anaconda" venv/pyvenv.cfg; then
        echo "Note: the existing venv was built on Conda's Python -- delete it once:  rm -rf venv"
    fi
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    echo "Error: No Python found. Please install Python 3." >&2
    exit 1
fi

# Create the venv if it doesn't exist yet
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment ($VENV_DIR) ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# Use the venv's own Python/pip path (no 'source activate' needed)
if [ -f "$VENV_DIR/bin/python" ]; then
    VENV_PYTHON="$VENV_DIR/bin/python"      # Linux/macOS
elif [ -f "$VENV_DIR/Scripts/python.exe" ]; then
    VENV_PYTHON="$VENV_DIR/Scripts/python.exe"   # Windows (Git Bash/WSL)
else
    echo "Error: Python not found inside the venv." >&2
    exit 1
fi

# Only (re)install dependencies if requirements.txt changed since the
# last run, or this is the first run.
if [ ! -f "$MARKER_FILE" ] || [ "requirements.txt" -nt "$MARKER_FILE" ]; then
    echo "Installing dependencies from requirements.txt ..."
    "$VENV_PYTHON" -m pip install --upgrade pip --quiet
    "$VENV_PYTHON" -m pip install -r requirements.txt --quiet
    touch "$MARKER_FILE"
else
    echo "Dependencies already installed, skipping pip install."
fi

echo "Starting Droplet Relaxer ..."
echo "--------------------------------------------------"

# Known issue: an active Conda environment sets LD_LIBRARY_PATH to its
# own bundled libraries (libstdc++, X11, etc.), which can shadow the
# system's Mesa/OpenGL libraries and break VTK's OpenGL window
# creation (symptoms: "Could not find a decent config"/"...visual",
# or a crash right after startup). Since this script's own environment
# variables never leak back into your shell, it's safe to strip
# LD_LIBRARY_PATH here rather than asking you to run
# 'conda deactivate' every time.
if [ -n "${CONDA_PREFIX:-}" ] && [ -n "${LD_LIBRARY_PATH:-}" ]; then
    echo "Detected an active Conda environment (CONDA_PREFIX=$CONDA_PREFIX)."
    echo "Unsetting LD_LIBRARY_PATH for this run only, to avoid a known"
    echo "conflict between Conda's bundled libraries and the system's"
    echo "OpenGL/Mesa libraries (see README.md -> Troubleshooting)."
    unset LD_LIBRARY_PATH
fi

# Optional escape hatch: software OpenGL (Mesa llvmpipe). Slower, but it
# works when the GPU driver is not loaded (typical after a reboot/kernel
# or driver update) and VTK reports "Could not find a decent config".
#   DROPLET_SOFTWARE_GL=1 ./run.sh
if [ "${DROPLET_SOFTWARE_GL:-0}" = "1" ]; then
    echo "DROPLET_SOFTWARE_GL=1: using software OpenGL (Mesa llvmpipe)."
    export LIBGL_ALWAYS_SOFTWARE=1
    export GALLIUM_DRIVER=llvmpipe
fi

# Quick, non-fatal look at the graphics setup, only to print a hint
# BEFORE VTK crashes with a segmentation fault.
if [ -z "${DISPLAY:-}" ]; then
    echo "Hint: DISPLAY is not set -- no X11/XWayland session reachable from this shell."
fi
if [ "${XDG_SESSION_TYPE:-}" = "wayland" ]; then
    echo "Hint: Wayland session detected. If the window fails to open, log in with"
    echo "      an 'Xorg' session instead, or try: DROPLET_SOFTWARE_GL=1 ./run.sh"
fi
if command -v glxinfo &>/dev/null; then
    if ! glxinfo -B >/dev/null 2>&1; then
        echo "Warning: 'glxinfo -B' fails -> OpenGL is not working outside Python either"
        echo "         (GPU driver not loaded?). Try: DROPLET_SOFTWARE_GL=1 ./run.sh"
    fi
fi

"$VENV_PYTHON" main.py "$@"
