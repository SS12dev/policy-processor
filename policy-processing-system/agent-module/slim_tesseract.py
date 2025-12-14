"""
Slim Tesseract Installation

Reduces Tesseract vendor directory size by keeping only essential language files.

Default: Keeps only English (eng) + orientation (osd)
Reduces size from ~800 MB to ~10 MB

Usage:
    python slim_tesseract.py                    # Keep English only
    python slim_tesseract.py --keep eng spa     # Keep English + Spanish
    python slim_tesseract.py --list             # List all available languages
"""

import sys
import logging
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

AGENT_MODULE_DIR = Path(__file__).parent
TESSDATA_DIR = AGENT_MODULE_DIR / "vendor" / "tesseract" / "tessdata"


def list_languages():
    """List all available language files"""
    if not TESSDATA_DIR.exists():
        logger.error(f"Tessdata directory not found: {TESSDATA_DIR}")
        logger.error("Run setup_dependencies.py first to bundle Tesseract")
        return

    traineddata_files = list(TESSDATA_DIR.glob("*.traineddata"))

    if not traineddata_files:
        logger.warning("No language files found")
        return

    logger.info(f"Found {len(traineddata_files)} language files:")
    logger.info("")

    # Group by type
    scripts = []
    languages = []
    other = []

    for f in sorted(traineddata_files):
        name = f.stem
        if name in ['osd', 'equ']:
            other.append(name)
        elif '_' in name:
            scripts.append(name)
        else:
            languages.append(name)

    if other:
        logger.info("Essential files:")
        for lang in other:
            logger.info(f"  - {lang}")
        logger.info("")

    if languages:
        logger.info(f"Languages ({len(languages)}):")
        # Display in columns
        for i in range(0, len(languages), 6):
            row = languages[i:i+6]
            logger.info(f"  {', '.join(row)}")
        logger.info("")

    if scripts:
        logger.info(f"Scripts/Special ({len(scripts)}):")
        for i in range(0, len(scripts), 4):
            row = scripts[i:i+4]
            logger.info(f"  {', '.join(row)}")


def slim_tesseract(keep_languages: list = None, dry_run: bool = False):
    """
    Remove unnecessary language files from Tesseract.

    Args:
        keep_languages: List of language codes to keep (e.g., ['eng', 'spa'])
        dry_run: If True, only show what would be deleted
    """
    if keep_languages is None:
        keep_languages = ['eng']  # Default: English only

    # Always keep these
    essential = ['osd', 'equ']  # Orientation & equation detection
    keep_languages = list(set(keep_languages + essential))

    if not TESSDATA_DIR.exists():
        logger.error(f"Tessdata directory not found: {TESSDATA_DIR}")
        logger.error("Run setup_dependencies.py first to bundle Tesseract")
        return False

    traineddata_files = list(TESSDATA_DIR.glob("*.traineddata"))

    if not traineddata_files:
        logger.warning("No language files found")
        return False

    # Determine what to keep and what to delete
    to_keep = []
    to_delete = []

    for f in traineddata_files:
        lang_code = f.stem
        if lang_code in keep_languages:
            to_keep.append(f)
        else:
            to_delete.append(f)

    # Calculate sizes
    total_size = sum(f.stat().st_size for f in traineddata_files)
    keep_size = sum(f.stat().st_size for f in to_keep)
    delete_size = sum(f.stat().st_size for f in to_delete)

    logger.info("=" * 80)
    logger.info("Tesseract Language File Cleanup")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Keeping languages: {', '.join(sorted(keep_languages))}")
    logger.info("")
    logger.info(f"Total files: {len(traineddata_files)}")
    logger.info(f"  Keep:   {len(to_keep)} files ({keep_size / (1024*1024):.1f} MB)")
    logger.info(f"  Delete: {len(to_delete)} files ({delete_size / (1024*1024):.1f} MB)")
    logger.info("")
    logger.info(f"Disk space savings: {delete_size / (1024*1024):.1f} MB")
    logger.info("")

    if dry_run:
        logger.info("DRY RUN - No files will be deleted")
        logger.info("")
        logger.info("Files that would be deleted:")
        for f in sorted(to_delete)[:20]:  # Show first 20
            logger.info(f"  - {f.name}")
        if len(to_delete) > 20:
            logger.info(f"  ... and {len(to_delete) - 20} more")
        logger.info("")
        logger.info("Run without --dry-run to actually delete files")
        return True

    # Confirm deletion
    response = input(f"Delete {len(to_delete)} language files? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        logger.info("Cancelled by user")
        return False

    # Delete files
    deleted_count = 0
    for f in to_delete:
        try:
            f.unlink()
            deleted_count += 1
        except Exception as e:
            logger.error(f"Failed to delete {f.name}: {e}")

    logger.info("")
    logger.info(f"✅ Deleted {deleted_count} files")
    logger.info(f"✅ Freed {delete_size / (1024*1024):.1f} MB of disk space")
    logger.info("")
    logger.info("Tesseract is now optimized for:")
    logger.info(f"  Languages: {', '.join(sorted([l for l in keep_languages if l not in essential]))}")
    logger.info("")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Slim Tesseract installation by removing unnecessary language files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python slim_tesseract.py                      # Keep English only (~10 MB)
  python slim_tesseract.py --keep eng spa fra   # Keep English, Spanish, French
  python slim_tesseract.py --list               # List all available languages
  python slim_tesseract.py --dry-run            # Preview what would be deleted

Common language codes:
  eng - English          spa - Spanish          fra - French
  deu - German           ita - Italian          por - Portuguese
  rus - Russian          ara - Arabic           chi_sim - Chinese Simplified
  jpn - Japanese         kor - Korean           hin - Hindi
        """
    )

    parser.add_argument(
        '--keep',
        nargs='+',
        metavar='LANG',
        help='Language codes to keep (e.g., eng spa fra)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available language files and exit'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )

    args = parser.parse_args()

    try:
        if args.list:
            list_languages()
            return

        keep_langs = args.keep if args.keep else ['eng']
        success = slim_tesseract(keep_languages=keep_langs, dry_run=args.dry_run)

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\n\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
