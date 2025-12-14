"""
Automatic Poppler installer for Windows.

Downloads and configures Poppler binaries for the agent module,
eliminating the need for manual installation.
"""

import os
import sys
import zipfile
import logging
from pathlib import Path
from typing import Optional
import urllib.request
import shutil

logger = logging.getLogger(__name__)

# Poppler Windows release URL
POPPLER_VERSION = "24.08.0-0"
POPPLER_DOWNLOAD_URL = f"https://github.com/oschwartz10612/poppler-windows/releases/download/v{POPPLER_VERSION}/Release-{POPPLER_VERSION}.zip"

# Local paths
AGENT_MODULE_DIR = Path(__file__).parent.parent
POPPLER_DIR = AGENT_MODULE_DIR / "vendor" / "poppler"
POPPLER_BIN_DIR = POPPLER_DIR / "Library" / "bin"


def is_poppler_installed() -> bool:
    """
    Check if Poppler is installed and accessible.

    Returns:
        True if Poppler is available, False otherwise
    """
    # Check if pdfinfo is accessible
    try:
        import subprocess
        result = subprocess.run(
            ["pdfinfo", "-v"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check if local vendor installation exists
    if POPPLER_BIN_DIR.exists():
        pdfinfo_path = POPPLER_BIN_DIR / "pdfinfo.exe"
        if pdfinfo_path.exists():
            return True

    return False


def download_poppler(force: bool = False) -> bool:
    """
    Download Poppler binaries for Windows.

    Args:
        force: Force re-download even if already exists

    Returns:
        True if successful, False otherwise
    """
    if not force and POPPLER_DIR.exists():
        logger.info(f"Poppler already downloaded at {POPPLER_DIR}")
        return True

    try:
        # Create vendor directory
        POPPLER_DIR.parent.mkdir(parents=True, exist_ok=True)

        # Download zip file
        zip_path = POPPLER_DIR.parent / "poppler.zip"
        logger.info(f"Downloading Poppler from {POPPLER_DOWNLOAD_URL}...")

        with urllib.request.urlopen(POPPLER_DOWNLOAD_URL) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192

            with open(zip_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Download progress: {progress:.1f}%")

        logger.info(f"Download complete: {downloaded / (1024*1024):.1f} MB")

        # Extract zip file
        logger.info(f"Extracting Poppler to {POPPLER_DIR}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract to temporary directory first
            temp_dir = POPPLER_DIR.parent / "poppler_temp"
            zip_ref.extractall(temp_dir)

            # Find the extracted directory (usually named "poppler-XX.XX.X")
            extracted_dirs = [d for d in temp_dir.iterdir() if d.is_dir()]
            if extracted_dirs:
                # Move contents to final location
                shutil.move(str(extracted_dirs[0]), str(POPPLER_DIR))
            else:
                # If no subdirectory, use temp_dir directly
                shutil.move(str(temp_dir), str(POPPLER_DIR))

            # Clean up temp directory if it still exists
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

        # Remove zip file
        zip_path.unlink()

        logger.info(f"Poppler installed successfully at {POPPLER_DIR}")
        return True

    except Exception as e:
        logger.error(f"Failed to download/extract Poppler: {e}")

        # Clean up on failure
        if zip_path.exists():
            zip_path.unlink()
        if POPPLER_DIR.exists():
            shutil.rmtree(POPPLER_DIR)

        return False


def configure_poppler_path() -> bool:
    """
    Add Poppler bin directory to system PATH for current process.

    Returns:
        True if successful, False otherwise
    """
    if not POPPLER_BIN_DIR.exists():
        logger.error(f"Poppler bin directory not found: {POPPLER_BIN_DIR}")
        return False

    try:
        # Add to PATH for current process
        poppler_bin_str = str(POPPLER_BIN_DIR.resolve())

        # Check if already in PATH
        current_path = os.environ.get('PATH', '')
        if poppler_bin_str not in current_path:
            os.environ['PATH'] = f"{poppler_bin_str}{os.pathsep}{current_path}"
            logger.info(f"Added Poppler to PATH: {poppler_bin_str}")
        else:
            logger.info("Poppler already in PATH")

        return True

    except Exception as e:
        logger.error(f"Failed to configure Poppler PATH: {e}")
        return False


def verify_poppler_installation() -> bool:
    """
    Verify that Poppler is properly installed and accessible.

    Returns:
        True if Poppler works correctly, False otherwise
    """
    try:
        import subprocess

        # Try to run pdfinfo -v
        result = subprocess.run(
            ["pdfinfo", "-v"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            version_output = result.stdout + result.stderr
            logger.info(f"Poppler verified successfully: {version_output.strip()}")
            return True
        else:
            logger.error(f"pdfinfo command failed with code {result.returncode}")
            return False

    except FileNotFoundError:
        logger.error("pdfinfo command not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        logger.error("pdfinfo command timed out")
        return False
    except Exception as e:
        logger.error(f"Failed to verify Poppler: {e}")
        return False


def ensure_poppler_available(auto_install: bool = True) -> bool:
    """
    Ensure Poppler is available for the agent module.

    This is the main entry point - call this on agent startup.

    Args:
        auto_install: Automatically download and install if not found

    Returns:
        True if Poppler is available, False otherwise
    """
    logger.info("Checking Poppler availability...")

    # Check if already installed globally
    if is_poppler_installed():
        logger.info("Poppler found in system PATH")
        # Verify it works
        if verify_poppler_installation():
            logger.info("Poppler verified successfully")
            return True
        else:
            logger.warning("Poppler found in PATH but verification failed")

    logger.info("Poppler not found in system PATH, checking local installation...")

    # Check if local vendor installation exists
    if POPPLER_BIN_DIR.exists():
        logger.info(f"Poppler found at {POPPLER_DIR}")

        # Configure PATH
        if configure_poppler_path():
            # Verify it works
            if verify_poppler_installation():
                logger.info("Poppler configured successfully from local installation")
                return True
            else:
                logger.warning("PATH configured but pdfinfo still not accessible")

        logger.warning("Local Poppler installation exists but is not working correctly")

    # Auto-install if requested
    if auto_install:
        logger.info("Attempting to auto-install Poppler...")

        if download_poppler():
            if configure_poppler_path():
                if verify_poppler_installation():
                    logger.info("Poppler auto-installation successful!")
                    return True

        logger.error("Poppler auto-installation failed")
        return False

    logger.error("Poppler not available and auto-install disabled")
    return False


def get_poppler_info() -> dict:
    """
    Get information about Poppler installation.

    Returns:
        Dictionary with installation details
    """
    return {
        "installed": is_poppler_installed(),
        "local_path": str(POPPLER_DIR) if POPPLER_DIR.exists() else None,
        "bin_path": str(POPPLER_BIN_DIR) if POPPLER_BIN_DIR.exists() else None,
        "in_system_path": is_poppler_installed() and not POPPLER_BIN_DIR.exists(),
        "version": POPPLER_VERSION,
    }


if __name__ == "__main__":
    # Test installation
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("Poppler Auto-Installer Test")
    print("=" * 80)

    info = get_poppler_info()
    print(f"\nCurrent Status:")
    print(f"  Installed: {info['installed']}")
    print(f"  Local Path: {info['local_path']}")
    print(f"  Bin Path: {info['bin_path']}")
    print(f"  In System PATH: {info['in_system_path']}")

    print(f"\nEnsuring Poppler is available...")
    success = ensure_poppler_available(auto_install=True)

    if success:
        print("\n✅ SUCCESS: Poppler is ready to use!")

        # Show final info
        info = get_poppler_info()
        print(f"\nFinal Status:")
        print(f"  Installed: {info['installed']}")
        print(f"  Local Path: {info['local_path']}")
        print(f"  Bin Path: {info['bin_path']}")
    else:
        print("\n❌ FAILED: Could not install Poppler")
        print("\nPlease install manually:")
        print("  1. Download: https://github.com/oschwartz10612/poppler-windows/releases/")
        print("  2. Extract to: C:\\Program Files\\poppler")
        print("  3. Add to PATH: C:\\Program Files\\poppler\\Library\\bin")

    print("=" * 80)
