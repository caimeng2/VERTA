#!/usr/bin/env python3
"""
Route Analyzer GUI Launcher
===========================

Simple launcher script for the Route Analyzer web GUI.
This script ensures proper imports and launches the Streamlit interface.

Usage:
    python launch_gui.py
    streamlit run launch_gui.py
"""

import sys
import os
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent / "project"
sys.path.insert(0, str(project_dir))

# Import and launch the GUI
try:
    from ra_gui import launch_gui
    
    if __name__ == "__main__":
        print("🚀 Launching Route Analyzer GUI...")
        print("📱 The web interface will open in your browser")
        print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
        print()
        
        launch_gui()
        
except ImportError as e:
    print("❌ Error importing GUI components:")
    print(f"   {e}")
    print()
    print("💡 Make sure you have installed the GUI dependencies:")
    print("   pip install -r requirements_gui.txt")
    print()
    print("💡 Also ensure you're running from the correct directory")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error launching GUI: {e}")
    sys.exit(1)
