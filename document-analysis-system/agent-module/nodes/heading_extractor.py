"""
Heading Extractor Node
Extracts headings and section titles from text.
"""

import re
from typing import Dict, Any, List, Tuple

from settings import settings
from utils.logger import get_logger
from utils.metrics import track_node_execution

logger = get_logger(__name__)


def extract_headings_basic(text: str, max_headings: int) -> List[Dict[str, Any]]:
    """
    Extract headings using pattern matching.

    Looks for:
    - Lines in ALL CAPS
    - Lines ending with colons
    - Numbered sections (1., 1.1, etc.)
    - Short lines followed by longer content

    Args:
        text: Input text
        max_headings: Maximum number of headings to extract

    Returns:
        List of heading dictionaries with level, text, and position
    """
    headings = []
    lines = text.split('\n')

    for i, line in enumerate(lines):
        line = line.strip()

        if not line or len(line) < 3:
            continue

        # Pattern 1: ALL CAPS headings (but not too long)
        if line.isupper() and 3 <= len(line) <= 100 and not re.search(r'\d{4}', line):
            headings.append({
                "level": 1,
                "text": line,
                "line_number": i + 1,
                "type": "caps"
            })
            continue

        # Pattern 2: Lines ending with colon
        if line.endswith(':') and len(line) <= 100:
            headings.append({
                "level": 2,
                "text": line.rstrip(':'),
                "line_number": i + 1,
                "type": "colon"
            })
            continue

        # Pattern 3: Numbered sections
        numbered_pattern = r'^(\d+\.(?:\d+\.)*)\s+(.+)$'
        match = re.match(numbered_pattern, line)
        if match:
            number, heading_text = match.groups()
            level = number.count('.') + 1
            headings.append({
                "level": min(level, 3),
                "text": heading_text,
                "line_number": i + 1,
                "type": "numbered",
                "number": number
            })
            continue

        # Pattern 4: Short lines (potential headings)
        # Must be between 10-80 chars, start with capital, no ending punctuation
        if (10 <= len(line) <= 80 and
            line[0].isupper() and
            not line[-1] in '.!?,;' and
            # Check if next line is longer (indicates heading)
            i + 1 < len(lines) and
            len(lines[i + 1].strip()) > len(line)):

            headings.append({
                "level": 3,
                "text": line,
                "line_number": i + 1,
                "type": "short_line"
            })

    # Limit and sort by line number
    headings = sorted(headings, key=lambda x: x["line_number"])[:max_headings]

    return headings


async def extract_headings_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract headings from extracted text.

    Args:
        state: Graph state containing extracted_text

    Returns:
        Updated state with headings list
    """
    with track_node_execution("heading_extractor"):
        try:
            logger.info("Starting heading extraction")

            extracted_text = state.get("extracted_text", "")

            if not extracted_text:
                logger.warning("No extracted text available for heading extraction")
                return {
                    **state,
                    "headings": [],
                    "heading_extraction_success": False
                }

            max_headings = settings.max_headings

            # Extract headings
            headings = extract_headings_basic(extracted_text, max_headings)

            logger.info(
                f"Extracted {len(headings)} headings",
                extra={
                    "by_type": {
                        htype: len([h for h in headings if h["type"] == htype])
                        for htype in set(h["type"] for h in headings)
                    }
                }
            )

            return {
                **state,
                "headings": headings,
                "heading_count": len(headings),
                "heading_extraction_success": True
            }

        except Exception as e:
            logger.error(
                f"Heading extraction failed: {str(e)}",
                extra={"error_type": type(e).__name__}
            )

            return {
                **state,
                "headings": [],
                "heading_count": 0,
                "heading_extraction_success": False,
                "heading_extraction_error": str(e)
            }
