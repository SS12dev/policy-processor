"""
Automatic Tesseract OCR installer for Windows.

Downloads and configures portable Tesseract OCR binaries for the agent module,
eliminating the need for manual installation.

Uses jTessBoxEditor portable builds which include Tesseract + tessdata.
"""

import os
import sys
import zipfile
import logging
from pathlib import Path
from typing import Optional
import urllib.request
import subprocess
import shutil

logger = logging.getLogger(__name__)

# Tesseract portable build URL (jTessBoxEditor includes portable Tesseract)
# Alternative: Use unofficial portable builds
TESSERACT_PORTABLE_URL = "https://github.com/tesseract-ocr/tesseract/releases/download/5.3.3/tesseract-v5.3.3.20231005-win64.zip"

# Fallback: Use a known working portable build
# This is a direct zip with tesseract.exe + eng.traineddata
TESSERACT_DOWNLOAD_URL = "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe"

# Local paths
AGENT_MODULE_DIR = Path(__file__).parent.parent
TESSERACT_DIR = AGENT_MODULE_DIR / "vendor" / "tesseract"
TESSERACT_EXE = TESSERACT_DIR / "tesseract.exe"
TESSDATA_DIR = TESSERACT_DIR / "tessdata"


def is_tesseract_installed() -> bool:
    """
    Check if Tesseract is installed and accessible.

    Returns:
        True if Tesseract is available, False otherwise
    """
    # Check if tesseract is accessible in PATH
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check if local vendor installation exists
    if TESSERACT_EXE.exists():
        return True

    # Check for system installation in common Windows paths
    common_paths = [
        Path("C:/Program Files/Tesseract-OCR"),
        Path("C:/Program Files (x86)/Tesseract-OCR"),
    ]
    for path in common_paths:
        if path.exists() and (path / "tesseract.exe").exists():
            return True

    return False


def download_and_extract_portable_tesseract(force: bool = False) -> bool:
    """
    Download and extract portable Tesseract build.

    Uses tesseract portable zip from official releases.

    Args:
        force: Force re-download even if already exists

    Returns:
        True if successful, False otherwise
    """
    if not force and TESSERACT_EXE.exists():
        logger.info(f"Tesseract already installed at {TESSERACT_DIR}")
        return True

    try:
        # Create vendor directory
        TESSERACT_DIR.parent.mkdir(parents=True, exist_ok=True)

        # Download portable zip
        zip_path = TESSERACT_DIR.parent / "tesseract.zip"
        logger.info(f"Downloading Tesseract portable from {TESSERACT_PORTABLE_URL}...")

        try:
            with urllib.request.urlopen(TESSERACT_PORTABLE_URL) as response:
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
                            if downloaded % (chunk_size * 100) == 0:  # Log every ~800KB
                                logger.info(f"Download progress: {progress:.1f}%")

            logger.info(f"Download complete: {downloaded / (1024*1024):.1f} MB")

        except Exception as download_error:
            logger.error(f"Failed to download from primary URL: {download_error}")
            logger.warning("Tesseract portable build not available for auto-download")
            logger.warning("Please install manually - see instructions below")

            # Clean up
            if zip_path.exists():
                zip_path.unlink()

            return False

        # Extract zip file
        logger.info(f"Extracting Tesseract to {TESSERACT_DIR}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TESSERACT_DIR)

        # Remove zip file
        zip_path.unlink()

        # Verify extraction
        if not TESSERACT_EXE.exists():
            logger.error("Tesseract.exe not found after extraction")
            logger.error("Extraction may have failed or directory structure is unexpected")
            return False

        # Create tessdata directory if it doesn't exist
        TESSDATA_DIR.mkdir(exist_ok=True)

        # Download English language data if not present
        eng_traineddata = TESSDATA_DIR / "eng.traineddata"
        if not eng_traineddata.exists():
            logger.info("Downloading English language data...")
            lang_url = "https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata"

            try:
                with urllib.request.urlopen(lang_url) as response:
                    with open(eng_traineddata, 'wb') as f:
                        f.write(response.read())
                logger.info("English language data downloaded")
            except Exception as e:
                logger.warning(f"Failed to download language data: {e}")
                logger.warning("OCR may not work without language data")

        logger.info(f"Tesseract installed successfully at {TESSERACT_DIR}")
        return True

    except Exception as e:
        logger.error(f"Failed to install Tesseract: {e}")

        # Clean up on failure
        if zip_path.exists():
            zip_path.unlink()
        if TESSERACT_DIR.exists():
            shutil.rmtree(TESSERACT_DIR)

        return False


def download_tesseract_portable(force: bool = False) -> bool:
    """
    Download portable Tesseract build for self-contained deployment.

    This creates a portable installation by:
    1. Downloading pre-compiled binaries from a reliable source
    2. Downloading English language data
    3. Setting up directory structure

    Args:
        force: Force re-download even if already exists

    Returns:
        True if successful, False otherwise
    """
    if not force and TESSERACT_EXE.exists():
        logger.info(f"Tesseract already installed at {TESSERACT_DIR}")
        return True

    try:
        # Create vendor directory
        TESSERACT_DIR.mkdir(parents=True, exist_ok=True)
        TESSDATA_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading portable Tesseract binaries...")

        # Download tesseract.exe from a reliable mirror
        # Using pre-built binaries from the Tesseract project
        tesseract_exe_url = "https://github.com/UB-Mannheim/tesseract/releases/download/v5.3.3.20231005/tesseract.exe"

        # Alternative: Use a direct download from a known location
        # For now, we'll guide the user to copy from system installation
        system_tesseract = find_system_tesseract()

        if system_tesseract:
            logger.info(f"Found system Tesseract at: {system_tesseract}")
            logger.info("Copying system Tesseract to vendor directory for portable deployment...")

            # Copy tesseract.exe
            system_exe = system_tesseract / "tesseract.exe"
            if system_exe.exists():
                import shutil
                shutil.copy2(system_exe, TESSERACT_EXE)
                logger.info(f"  ✅ Copied tesseract.exe")

            # Copy all DLL dependencies
            for dll in system_tesseract.glob("*.dll"):
                shutil.copy2(dll, TESSERACT_DIR / dll.name)
                logger.debug(f"  Copied {dll.name}")

            # Copy tessdata directory
            system_tessdata = system_tesseract / "tessdata"
            if system_tessdata.exists():
                import shutil
                if TESSDATA_DIR.exists():
                    shutil.rmtree(TESSDATA_DIR)
                shutil.copytree(system_tessdata, TESSDATA_DIR)
                logger.info(f"  ✅ Copied tessdata directory ({len(list(TESSDATA_DIR.glob('*.traineddata')))} language files)")

            logger.info(f"✅ Portable Tesseract created successfully at {TESSERACT_DIR}")
            return True

        else:
            logger.warning("No system Tesseract installation found to copy from")
            logger.warning("")
            logger.warning("To create a portable Tesseract installation:")
            logger.warning("1. Install Tesseract from: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe")
            logger.warning("2. Run this setup script again")
            logger.warning("   It will copy the system installation to create a portable version")
            logger.warning("")
            return False

    except Exception as e:
        logger.error(f"Failed to create portable Tesseract: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_system_tesseract() -> Optional[Path]:
    """
    Find Tesseract installation in common Windows locations.

    Returns:
        Path to Tesseract directory if found, None otherwise
    """
    common_paths = [
        Path("C:/Program Files/Tesseract-OCR"),
        Path("C:/Program Files (x86)/Tesseract-OCR"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR",
        Path(os.environ.get("PROGRAMFILES", "")) / "Tesseract-OCR",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Tesseract-OCR",
    ]

    for path in common_paths:
        if path.exists() and (path / "tesseract.exe").exists():
            logger.info(f"Found system Tesseract at: {path}")
            return path

    return None


def configure_tesseract_path() -> bool:
    """
    Add Tesseract directory to system PATH for current process.

    Checks both local vendor installation and system installation.

    Returns:
        True if successful, False otherwise
    """
    tesseract_path = None

    # First, check vendor installation
    if TESSERACT_DIR.exists() and TESSERACT_EXE.exists():
        tesseract_path = TESSERACT_DIR
        logger.debug(f"Using vendor Tesseract: {tesseract_path}")
    else:
        # Check for system installation
        tesseract_path = find_system_tesseract()
        if tesseract_path:
            logger.info(f"Using system Tesseract installation: {tesseract_path}")

    if not tesseract_path:
        logger.debug("No Tesseract installation found")
        return False

    try:
        # Add to PATH for current process
        tesseract_dir_str = str(tesseract_path.resolve())

        # Check if already in PATH
        current_path = os.environ.get('PATH', '')
        if tesseract_dir_str not in current_path:
            os.environ['PATH'] = f"{tesseract_dir_str}{os.pathsep}{current_path}"
            logger.info(f"Added Tesseract to PATH: {tesseract_dir_str}")
        else:
            logger.debug("Tesseract already in PATH")

        # Set TESSDATA_PREFIX environment variable
        tessdata_path = tesseract_path / "tessdata"
        if tessdata_path.exists():
            os.environ['TESSDATA_PREFIX'] = str(tessdata_path.resolve())
            logger.info(f"Set TESSDATA_PREFIX: {tessdata_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to configure Tesseract PATH: {e}")
        return False


def verify_tesseract_installation() -> bool:
    """
    Verify that Tesseract is properly installed and accessible.

    Returns:
        True if Tesseract works correctly, False otherwise
    """
    try:
        # Try to run tesseract --version
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            version_output = result.stdout + result.stderr
            # Extract version from output
            version_line = version_output.split('\n')[0] if version_output else "Unknown"
            logger.info(f"Tesseract verified: {version_line}")
            return True
        else:
            logger.error(f"tesseract command failed with code {result.returncode}")
            return False

    except FileNotFoundError:
        logger.debug("tesseract command not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        logger.error("tesseract command timed out")
        return False
    except Exception as e:
        logger.error(f"Failed to verify Tesseract: {e}")
        return False


def ensure_tesseract_available(auto_install: bool = True) -> bool:
    """
    Ensure Tesseract is available for the agent module.

    Args:
        auto_install: Automatically download and install if not found

    Returns:
        True if Tesseract is available, False otherwise
    """
    logger.info("Checking Tesseract OCR availability...")

    # Check if already installed globally
    if is_tesseract_installed():
        logger.info("Tesseract found in system PATH")

        # Verify it works
        if verify_tesseract_installation():
            return True

    logger.info("Tesseract not found in system PATH, checking for system installation...")

    # Check for system installation (e.g., C:\Program Files\Tesseract-OCR)
    system_tesseract = find_system_tesseract()
    if system_tesseract:
        logger.info(f"Found system Tesseract installation at: {system_tesseract}")

        # Configure PATH to include system installation
        if configure_tesseract_path():
            # Verify it works
            if verify_tesseract_installation():
                logger.info("✅ Tesseract configured successfully from system installation")
                return True

    logger.info("Checking for local vendor installation...")

    # Check if local vendor installation exists
    if TESSERACT_DIR.exists() and TESSERACT_EXE.exists():
        logger.info(f"Tesseract found at {TESSERACT_DIR}")

        # Configure PATH
        if configure_tesseract_path():
            # Verify it works
            if verify_tesseract_installation():
                logger.info("Tesseract configured successfully from local installation")
                return True

        logger.warning("Local Tesseract installation exists but is not working correctly")

    # Auto-install if requested
    if auto_install:
        logger.info("Attempting to auto-install Tesseract...")

        if download_and_extract_portable_tesseract():
            if configure_tesseract_path():
                if verify_tesseract_installation():
                    logger.info("Tesseract auto-installation successful!")
                    return True

        logger.error("Tesseract auto-installation failed")
        logger.warning("Falling back to manual installation instructions...")

    # Tesseract not found - provide installation instructions
    logger.warning("=" * 80)
    logger.warning("TESSERACT OCR NOT INSTALLED")
    logger.warning("=" * 80)
    logger.warning("")
    logger.warning("Tesseract is required for OCR (reading text from scanned PDFs)")
    logger.warning("")
    logger.warning("Quick Install Options:")
    logger.warning("")
    logger.warning("Option 1: Chocolatey (Recommended - Most Reliable)")
    logger.warning("  choco install tesseract")
    logger.warning("")
    logger.warning("Option 2: Scoop")
    logger.warning("  scoop install tesseract")
    logger.warning("")
    logger.warning("Option 3: Manual Download")
    logger.warning("  Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    logger.warning(f"  Direct link: {TESSERACT_DOWNLOAD_URL}")
    logger.warning("  During installation, check 'Add to PATH'")
    logger.warning("")
    logger.warning("After installation:")
    logger.warning("  1. Close and reopen your terminal")
    logger.warning("  2. Verify: tesseract --version")
    logger.warning("  3. Restart agent server")
    logger.warning("")
    logger.warning("Without Tesseract:")
    logger.warning("  - OCR will fail on scanned pages")
    logger.warning("  - Text-based PDFs will still work (using PyPDF2)")
    logger.warning("  - Policy extraction quality will be reduced for scanned documents")
    logger.warning("")
    logger.warning("=" * 80)

    return False


def get_tesseract_info() -> dict:
    """
    Get information about Tesseract installation.

    Returns:
        Dictionary with installation details
    """
    installed = is_tesseract_installed()

    version = None
    if installed:
        try:
            result = subprocess.run(
                ["tesseract", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                output = result.stdout + result.stderr
                # Extract version (first line usually has "tesseract X.Y.Z")
                version = output.split('\n')[0] if output else None
        except:
            pass

    return {
        "installed": installed,
        "version": version,
        "local_path": str(TESSERACT_DIR) if TESSERACT_DIR.exists() else None,
        "exe_path": str(TESSERACT_EXE) if TESSERACT_EXE.exists() else None,
        "tessdata_path": str(TESSDATA_DIR) if TESSDATA_DIR.exists() else None,
        "download_url": TESSERACT_PORTABLE_URL,
    }


if __name__ == "__main__":
    # Test installation
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("Tesseract OCR Auto-Installer Test")
    print("=" * 80)

    info = get_tesseract_info()
    print(f"\nCurrent Status:")
    print(f"  Installed: {info['installed']}")
    print(f"  Version: {info['version'] or 'N/A'}")
    print(f"  Local Path: {info['local_path'] or 'N/A'}")

    print(f"\nEnsuring Tesseract is available (auto-install enabled)...")
    available = ensure_tesseract_available(auto_install=True)

    if available:
        print("\n✅ SUCCESS: Tesseract is installed and ready!")

        # Show final info
        info = get_tesseract_info()
        print(f"\nFinal Status:")
        print(f"  Installed: {info['installed']}")
        print(f"  Version: {info['version']}")
        print(f"  Local Path: {info['local_path']}")
    else:
        print("\n❌ FAILED: Could not install Tesseract")
        print("\nPlease install manually (see instructions above)")

    print("=" * 80)
