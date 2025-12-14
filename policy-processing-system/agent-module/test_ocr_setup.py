"""
Simple OCR Setup Test Script

Tests Poppler and Tesseract installation without requiring Redis or other dependencies.
"""

import sys
import subprocess
import logging
from pathlib import Path

# Add agent-module to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def test_poppler():
    """Test Poppler installation"""
    logger.info("=" * 80)
    logger.info("Testing Poppler Installation")
    logger.info("=" * 80)

    try:
        from utils.poppler_installer import (
            is_poppler_installed,
            ensure_poppler_available,
            verify_poppler_installation,
            get_poppler_info
        )

        # Get current status
        info = get_poppler_info()
        logger.info(f"Current Status:")
        logger.info(f"  Installed: {info['installed']}")
        logger.info(f"  Version: {info['version'] or 'N/A'}")
        logger.info(f"  Local Path: {info['local_path'] or 'N/A'}")

        # Ensure available (auto-install if needed)
        logger.info("\nEnsuring Poppler is available (auto-install enabled)...")
        if ensure_poppler_available(auto_install=True):
            logger.info("✅ Poppler is available and working!")

            # Verify
            if verify_poppler_installation():
                logger.info("✅ Poppler verification: PASSED")
                return True
            else:
                logger.error("❌ Poppler verification: FAILED")
                return False
        else:
            logger.error("❌ Poppler installation: FAILED")
            return False

    except Exception as e:
        logger.error(f"❌ Error testing Poppler: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tesseract():
    """Test Tesseract installation"""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Tesseract Installation")
    logger.info("=" * 80)

    try:
        from utils.tesseract_installer import (
            is_tesseract_installed,
            ensure_tesseract_available,
            get_tesseract_info
        )

        # Get current status
        info = get_tesseract_info()
        logger.info(f"Current Status:")
        logger.info(f"  Installed: {info['installed']}")
        logger.info(f"  Version: {info['version'] or 'N/A'}")
        logger.info(f"  Local Path: {info['local_path'] or 'N/A'}")

        # Check and configure installation (this will auto-detect system installation and configure PATH)
        logger.info("\nEnsuring Tesseract is available (will configure PATH if needed)...")
        if ensure_tesseract_available(auto_install=False):
            logger.info("✅ Tesseract is available and working!")
            return True
        else:
            logger.warning("⚠️  Tesseract not found or failed to configure")
            logger.warning("\nTo install Tesseract:")
            logger.warning("  1. Download from: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe")
            logger.warning("  2. Run installer and check 'Add to PATH'")
            logger.warning("  3. Restart terminal")
            logger.warning("  4. Run this test again")
            return False

    except Exception as e:
        logger.error(f"❌ Error testing Tesseract: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ocr_pipeline():
    """Test complete OCR pipeline"""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Complete OCR Pipeline")
    logger.info("=" * 80)

    try:
        import pytesseract
        from pdf2image import convert_from_path

        logger.info("✅ pdf2image imported successfully")
        logger.info("✅ pytesseract imported successfully")

        # Note: Actual PDF conversion would require a test PDF file
        logger.info("\n📝 Note: Full OCR test requires a PDF file")
        logger.info("   Once both Poppler and Tesseract are installed,")
        logger.info("   you can test with actual documents via the agent server.")

        return True

    except ImportError as e:
        logger.error(f"❌ Missing Python package: {e}")
        logger.info("\nInstall missing packages:")
        logger.info("  pip install pdf2image pytesseract")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing OCR pipeline: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("\n")
    logger.info("🔧 OCR Setup Test Suite")
    logger.info("Testing Poppler and Tesseract installation for the agent module")
    logger.info("\n")

    results = {
        "poppler": test_poppler(),
        "tesseract": test_tesseract(),
        "pipeline": test_ocr_pipeline()
    }

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)
    logger.info(f"Poppler:    {'✅ PASS' if results['poppler'] else '❌ FAIL'}")
    logger.info(f"Tesseract:  {'✅ PASS' if results['tesseract'] else '❌ FAIL'}")
    logger.info(f"Pipeline:   {'✅ PASS' if results['pipeline'] else '❌ FAIL'}")
    logger.info("=" * 80)

    if all(results.values()):
        logger.info("\n🎉 SUCCESS: Full OCR pipeline is ready!")
        logger.info("\nNext steps:")
        logger.info("  1. Start agent server: cd agent-module && python server.py")
        logger.info("  2. Start client UI: cd client-module && streamlit run app.py")
        logger.info("  3. Process a document to verify OCR works end-to-end")
        return 0
    else:
        logger.info("\n⚠️  INCOMPLETE: Some components need attention")
        if not results['poppler']:
            logger.info("  - Fix Poppler installation")
        if not results['tesseract']:
            logger.info("  - Install Tesseract (see instructions above)")
        if not results['pipeline']:
            logger.info("  - Install Python packages: pip install pdf2image pytesseract")
        return 1


if __name__ == "__main__":
    sys.exit(main())
