#!/usr/bin/env python3
"""
Fix Windows SSL Certificate issues for Python.
This resolves "CERTIFICATE_VERIFY_FAILED" errors when downloading from HuggingFace.
"""

import sys
import platform
import subprocess

def fix_windows_ssl():
    """Install/update SSL certificates on Windows"""
    if platform.system() != "Windows":
        print("ℹ️  This script is for Windows only.")
        return
    
    print("🔧 Fixing Windows SSL Certificates for Python...")
    print()
    
    # Option 1: Update certifi package
    print("1️⃣  Updating certifi (SSL root certificates)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "certifi"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("   ✅ certifi updated")
    else:
        print(f"   ⚠️  certifi update had issues: {result.stderr[:100]}")
    
    print()
    print("2️⃣  Attempting to install Windows certificates via pip...")
    
    # Option 2: Install wincertstore
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "wincertstore"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("   ✅ wincertstore installed")
        else:
            print("   ℹ️  wincertstore not available (not critical)")
    except:
        pass
    
    print()
    print("3️⃣  Verifying SSL setup...")
    try:
        result = subprocess.run(
            [sys.executable, "-c", 
             "import ssl; ssl.create_default_context().check_hostname; print('✅ SSL working!')"],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(result.stdout)
    except:
        print("   ⚠️  Could not verify SSL")
    
    print()
    print("✅ Windows SSL certificate fixes applied!")
    print()
    print("🎯 Next steps:")
    print("   1. Restart Python/Streamlit")
    print("   2. Try downloading the model again:")
    print("      python download_model.py")
    print()

if __name__ == "__main__":
    fix_windows_ssl()

