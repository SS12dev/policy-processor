"""
Keyword Extractor Node
Extracts keywords using both basic NLP and LLM-based methods.
"""

import re
from typing import Dict, Any, List
from collections import Counter

from settings import settings
from utils.logger import get_logger
from utils.metrics import track_node_execution
from utils.llm import get_llm_client

logger = get_logger(__name__)


# Common English stop words
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
    'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how', 'all',
    'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such'
}


def extract_keywords_basic(text: str, max_keywords: int) -> List[str]:
    """
    Extract keywords using frequency-based analysis.

    Args:
        text: Input text
        max_keywords: Maximum number of keywords

    Returns:
        List of keywords
    """
    # Lowercase and extract words
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())

    # Filter stop words
    words = [w for w in words if w not in STOP_WORDS]

    # Count frequencies
    word_freq = Counter(words)

    # Also extract 2-word phrases
    words_list = text.lower().split()
    bigrams = [
        f"{words_list[i]} {words_list[i+1]}"
        for i in range(len(words_list) - 1)
        if words_list[i] not in STOP_WORDS and words_list[i+1] not in STOP_WORDS
    ]
    bigram_freq = Counter(bigrams)

    # Combine single words and bigrams
    all_keywords = []

    # Add top bigrams (phrases are often more meaningful)
    for phrase, count in bigram_freq.most_common(max_keywords // 2):
        if count >= 2:  # Must appear at least twice
            all_keywords.append(phrase)

    # Add top single words
    for word, count in word_freq.most_common(max_keywords):
        if count >= 3 and len(word) > 4:  # Must appear at least 3 times and be meaningful
            all_keywords.append(word)

    # Deduplicate and limit
    seen = set()
    unique_keywords = []
    for kw in all_keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)

    return unique_keywords[:max_keywords]


async def extract_keywords_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract keywords from text using hybrid approach (basic + LLM).

    Args:
        state: Graph state containing extracted_text

    Returns:
        Updated state with keywords
    """
    with track_node_execution("keyword_extractor"):
        try:
            logger.info("Starting keyword extraction")

            extracted_text = state.get("extracted_text", "")

            if not extracted_text:
                logger.warning("No extracted text available for keyword extraction")
                return {
                    **state,
                    "keywords": [],
                    "keyword_extraction_success": False
                }

            max_keywords = settings.max_keywords

            # Step 1: Basic keyword extraction
            basic_keywords = extract_keywords_basic(extracted_text, max_keywords)

            logger.info(
                f"Extracted {len(basic_keywords)} keywords using basic method",
                extra={"keywords_sample": basic_keywords[:5]}
            )

            # Step 2: LLM-enhanced extraction (if enabled)
            final_keywords = basic_keywords
            llm_keywords = []

            if settings.enable_llm_analysis:
                try:
                    llm_client = get_llm_client()

                    # Use a shorter text sample for LLM (to save tokens)
                    text_sample = extracted_text[:settings.chunk_size]

                    llm_keywords = await llm_client.extract_keywords(
                        text=text_sample,
                        max_keywords=max_keywords
                    )

                    logger.info(
                        f"Extracted {len(llm_keywords)} keywords using LLM",
                        extra={"llm_keywords_sample": llm_keywords[:5]}
                    )

                    # Merge basic and LLM keywords (prioritize LLM, add basic as fallback)
                    combined = []
                    seen = set()

                    for kw in llm_keywords:
                        if kw.lower() not in seen:
                            combined.append(kw)
                            seen.add(kw.lower())

                    for kw in basic_keywords:
                        if kw.lower() not in seen:
                            combined.append(kw)
                            seen.add(kw.lower())

                    final_keywords = combined[:max_keywords]

                except Exception as e:
                    logger.warning(
                        f"LLM keyword extraction failed, using basic keywords: {str(e)}"
                    )
                    final_keywords = basic_keywords

            logger.info(
                f"Keyword extraction completed with {len(final_keywords)} keywords",
                extra={"final_keywords": final_keywords}
            )

            return {
                **state,
                "keywords": final_keywords,
                "keywords_basic": basic_keywords,
                "keywords_llm": llm_keywords,
                "keyword_count": len(final_keywords),
                "keyword_extraction_success": True
            }

        except Exception as e:
            logger.error(
                f"Keyword extraction failed: {str(e)}",
                extra={"error_type": type(e).__name__}
            )

            return {
                **state,
                "keywords": [],
                "keyword_count": 0,
                "keyword_extraction_success": False,
                "keyword_extraction_error": str(e)
            }
