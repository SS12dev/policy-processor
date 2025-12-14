"""
Test script for Poppler auto-installation.

Run this to verify that Poppler auto-install works correctly.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.poppler_installer import (
    ensure_poppler_available,
    get_poppler_info,
    verify_poppler_installation
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Test Poppler installation."""
    print("=" * 80)
    print("Poppler Auto-Installation Test")
    print("=" * 80)

    # Show current status
    print("\n[Step 1] Checking current Poppler status...")
    info = get_poppler_info()
    print(f"  Installed: {info['installed']}")
    print(f"  Local Path: {info['local_path'] or 'None'}")
    print(f"  Bin Path: {info['bin_path'] or 'None'}")
    print(f"  In System PATH: {info['in_system_path']}")
    print(f"  Expected Version: {info['version']}")

    # Ensure Poppler is available
    print("\n[Step 2] Ensuring Poppler is available (auto-install if needed)...")
    success = ensure_poppler_available(auto_install=True)

    if not success:
        print("\n❌ FAILED: Could not ensure Poppler availability")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Check if download URL is accessible:")
        print("     https://github.com/oschwartz10612/poppler-windows/releases/")
        print("  3. Try manual installation (see POPPLER_INSTALLATION.md)")
        return 1

    print("\n✅ Poppler is available!")

    # Verify installation
    print("\n[Step 3] Verifying Poppler installation...")
    verified = verify_poppler_installation()

    if not verified:
        print("\n⚠️  WARNING: Poppler is installed but verification failed")
        print("  This might indicate a configuration issue")
        return 1

    print("\n✅ Poppler verified successfully!")

    # Show final status
    print("\n[Step 4] Final Poppler status...")
    info = get_poppler_info()
    print(f"  Installed: {info['installed']}")
    print(f"  Local Path: {info['local_path']}")
    print(f"  Bin Path: {info['bin_path']}")
    print(f"  In System PATH: {info['in_system_path']}")

    # Test with pdf2image
    print("\n[Step 5] Testing pdf2image integration...")
    try:
        from pdf2image import pdfinfo_from_path
        print("  ✅ pdf2image import successful")

        # Try to get info from a dummy call (will fail but shows Poppler is accessible)
        try:
            # This will fail but that's OK - we just want to see if Poppler is found
            pdfinfo_from_path("dummy.pdf")
        except Exception as e:
            error_msg = str(e)
            if "poppler" in error_msg.lower() or "path" in error_msg.lower():
                print(f"  ❌ pdf2image cannot find Poppler: {error_msg}")
                return 1
            else:
                # Other errors (like file not found) are expected and OK
                print(f"  ✅ pdf2image can access Poppler (error is expected: {type(e).__name__})")

    except ImportError:
        print("  ⚠️  WARNING: pdf2image not installed")
        print("     Install with: pip install pdf2image")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✅ Poppler auto-installation: SUCCESS")
    print("✅ Poppler verification: SUCCESS")
    print("✅ Ready for OCR processing")
    print("\nYou can now start the agent server:")
    print("  cd agent-module")
    print("  python server.py")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
