#!/usr/bin/env python3
"""
Route Analyzer GUI Launcher
===========================

Launch the Streamlit web UI reliably from either the repository root or the
`project/` directory.

Usage (from repo root):
  python -m project.launch_gui
  # or
  python project/launch_gui.py

Usage (from project/ directory):
  python launch_gui.py
"""

import os
import sys
import subprocess
from pathlib import Path


def main() -> int:
    # Determine repository root (one level up when running inside project/)
    this_file = Path(__file__).resolve()
    project_dir = this_file.parent
    repo_root = project_dir.parent

    # Path to GUI app
    gui_path = project_dir / "ra_gui.py"
    if not gui_path.exists():
        print(f"❌ GUI entry not found at: {gui_path}")
        return 1

    # Ensure Streamlit is available
    try:
        import streamlit  # noqa: F401
    except Exception:
        print("❌ Streamlit is not installed.")
        print("💡 Install GUI deps first: pip install -r requirements_gui.txt")
        return 1

    # Build streamlit command
    cmd = [sys.executable, "-m", "streamlit", "run", str(gui_path)]

    # Prefer running with repository root as CWD so absolute imports work too
    cwd = repo_root if repo_root.exists() else project_dir

    print("🚀 Launching Route Analyzer GUI...")
    print("   If the browser doesn't open automatically, visit: http://localhost:8501\n")

    # Inherit env and run
    env = os.environ.copy()
    try:
        result = subprocess.run(cmd, cwd=str(cwd))
        return result.returncode
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"❌ Failed to launch GUI: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
