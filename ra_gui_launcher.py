#!/usr/bin/env python3
"""
Route Analyzer GUI Launcher
===========================

Standalone launcher for the Route Analyzer web GUI.
This allows running the GUI without importing the package.

Usage:
    python ra_gui_launcher.py
    streamlit run ra_gui_launcher.py
"""

import sys
import os
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🗺️ Route Analyzer GUI Launcher")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Get the path to the GUI file
    current_dir = Path(__file__).parent
    gui_file = current_dir / "project" / "ra_gui.py"
    
    if not gui_file.exists():
        print(f"❌ GUI file not found: {gui_file}")
        sys.exit(1)
    
    print("✅ Dependencies check passed")
    print("🚀 Starting Route Analyzer GUI...")
    print("\n📝 The GUI will open in your web browser")
    print("   If it doesn't open automatically, go to: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("-" * 40)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(gui_file),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Route Analyzer GUI stopped")
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
