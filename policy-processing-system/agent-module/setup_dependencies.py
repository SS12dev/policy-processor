"""
Setup Dependencies Script

Downloads and configures all OCR dependencies (Poppler + Tesseract) for the agent module.
Run this once during deployment to create a fully self-contained installation.

Usage:
    python setup_dependencies.py

This will:
1. Download Poppler portable binaries (~35 MB download, ~90 MB extracted)
2. Download Tesseract portable package (~5 MB download, ~60 MB extracted)
3. Configure both for the agent module
4. Verify installations work correctly

After running this, the agent module will be completely self-contained.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def setup_all_dependencies(force: bool = False) -> bool:
    """
    Download and configure all dependencies.

    Args:
        force: Force re-download even if already exists

    Returns:
        True if all dependencies are ready, False otherwise
    """
    logger.info("=" * 80)
    logger.info("Policy Processing System - Dependency Setup")
    logger.info("=" * 80)
    logger.info("")
    logger.info("This will download and configure:")
    logger.info("  - Poppler utilities (~90 MB)")
    logger.info("  - Tesseract OCR (~60 MB)")
    logger.info("")
    logger.info("Total download size: ~40 MB")
    logger.info("Total disk space: ~150 MB")
    logger.info("")

    if not force:
        response = input("Continue? [Y/n]: ").strip().lower()
        if response and response not in ['y', 'yes']:
            logger.info("Setup cancelled by user")
            return False

    logger.info("")
    logger.info("Starting dependency setup...")
    logger.info("")

    success = True

    # Setup Poppler
    logger.info("[1/2] Setting up Poppler...")
    logger.info("-" * 80)
    try:
        from utils.poppler_installer import (
            download_poppler,
            configure_poppler_path,
            verify_poppler_installation
        )

        if download_poppler(force=force):
            logger.info("  ✅ Poppler downloaded successfully")

            if configure_poppler_path():
                logger.info("  ✅ Poppler PATH configured")

                if verify_poppler_installation():
                    logger.info("  ✅ Poppler verified and working")
                else:
                    logger.error("  ❌ Poppler verification failed")
                    success = False
            else:
                logger.error("  ❌ Poppler PATH configuration failed")
                success = False
        else:
            logger.error("  ❌ Poppler download failed")
            success = False

    except Exception as e:
        logger.error(f"  ❌ Error setting up Poppler: {e}")
        success = False

    logger.info("")

    # Setup Tesseract
    logger.info("[2/2] Setting up Tesseract...")
    logger.info("-" * 80)
    try:
        from utils.tesseract_installer import (
            download_tesseract_portable,
            configure_tesseract_path,
            verify_tesseract_installation
        )

        if download_tesseract_portable(force=force):
            logger.info("  ✅ Tesseract downloaded successfully")

            if configure_tesseract_path():
                logger.info("  ✅ Tesseract PATH configured")

                if verify_tesseract_installation():
                    logger.info("  ✅ Tesseract verified and working")
                else:
                    logger.error("  ❌ Tesseract verification failed")
                    success = False
            else:
                logger.error("  ❌ Tesseract PATH configuration failed")
                success = False
        else:
            logger.warning("  ⚠️  Tesseract portable download not available")
            logger.warning("  ℹ️  Will use system installation if available")

    except Exception as e:
        logger.error(f"  ❌ Error setting up Tesseract: {e}")
        logger.warning("  ℹ️  Will fallback to system installation")

    logger.info("")
    logger.info("=" * 80)

    if success:
        logger.info("✅ SUCCESS: All dependencies are ready!")
        logger.info("")
        logger.info("Your agent module is now fully self-contained.")
        logger.info("You can deploy this directory to any Windows machine.")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Configure .env file with your OPENAI_API_KEY")
        logger.info("  2. Start agent: python server.py")
        logger.info("  3. Start client: cd ../client-module && streamlit run app.py")
        logger.info("")
    else:
        logger.warning("⚠️  PARTIAL SUCCESS: Some dependencies may not be available")
        logger.warning("")
        logger.warning("The agent will still work with:")
        logger.warning("  - System-installed Tesseract (if available)")
        logger.warning("  - Downloaded Poppler (if successful)")
        logger.warning("")
        logger.warning("For full self-contained deployment, install missing dependencies manually.")
        logger.warning("")

    logger.info("=" * 80)

    return success


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup OCR dependencies for Policy Processing Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_dependencies.py              # Interactive setup
  python setup_dependencies.py --force      # Force re-download
  python setup_dependencies.py --yes        # Non-interactive (auto-accept)

This creates a self-contained agent module that can be deployed anywhere.
        """
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if dependencies already exist'
    )

    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt (non-interactive mode)'
    )

    args = parser.parse_args()

    try:
        success = setup_all_dependencies(force=args.force or args.yes)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
