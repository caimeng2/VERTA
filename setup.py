#!/usr/bin/env python3
"""
Route Analyzer Setup Script
===========================

Easy setup script for Route Analyzer with optional GUI support.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package():
    """Install the route analyzer package"""
    print("📦 Installing Route Analyzer package...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print("✅ Package installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Package installation failed: {e}")
        return False

def install_gui_dependencies():
    """Install GUI dependencies"""
    print("🖥️ Installing GUI dependencies...")
    try:
        requirements_file = Path(__file__).parent / "requirements_gui.txt"
        if requirements_file.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], check=True)
            print("✅ GUI dependencies installed successfully")
            return True
        else:
            print("⚠️ GUI requirements file not found")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ GUI dependencies installation failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🗺️ Route Analyzer Setup")
    print("=" * 30)
    
    # Install core package
    if not install_package():
        sys.exit(1)
    
    # Ask about GUI installation
    install_gui = input("\n🖥️ Install GUI dependencies? (y/n): ").lower().strip()
    
    if install_gui in ['y', 'yes']:
        if not install_gui_dependencies():
            print("⚠️ GUI installation failed, but core package is ready")
    
    print("\n🎉 Setup complete!")
    print("\n📖 Usage:")
    print("   CLI Mode:    python -m project.route_analyzer <command>")
    print("   GUI Mode:     python -m project.route_analyzer --gui")
    print("   GUI Launcher: python ra_gui_launcher.py")
    print("\n💡 For help: python -m project.route_analyzer --help")

if __name__ == "__main__":
    main()
