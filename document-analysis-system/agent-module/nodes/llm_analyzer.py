"""
LLM Analyzer Node
Uses LLM to perform deep analysis of the document.
"""

from typing import Dict, Any

from settings import settings
from utils.logger import get_logger
from utils.metrics import track_node_execution
from utils.llm import get_llm_client

logger = get_logger(__name__)


async def analyze_document_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform LLM-based document analysis.

    Args:
        state: Graph state with extracted_text, headings, and keywords

    Returns:
        Updated state with analysis results
    """
    with track_node_execution("llm_analyzer"):
        try:
            if not settings.enable_llm_analysis:
                logger.info("LLM analysis disabled, skipping")
                return {
                    **state,
                    "analysis": None,
                    "analysis_success": False
                }

            logger.info("Starting LLM document analysis")

            extracted_text = state.get("extracted_text", "")
            headings = state.get("headings", [])
            keywords = state.get("keywords", [])

            if not extracted_text:
                logger.warning("No text available for analysis")
                return {
                    **state,
                    "analysis": None,
                    "analysis_success": False
                }

            # Get LLM client
            llm_client = get_llm_client()

            # Perform analysis
            analysis = await llm_client.analyze_document(
                text=extracted_text,
                headings=[h["text"] for h in headings],
                keywords=keywords
            )

            logger.info(
                "LLM analysis completed",
                extra={
                    "summary_length": len(analysis.get("summary", "")),
                    "topics_count": len(analysis.get("main_topics", [])),
                    "document_type": analysis.get("document_type"),
                    "complexity": analysis.get("complexity_level")
                }
            )

            return {
                **state,
                "analysis": analysis,
                "analysis_success": True
            }

        except Exception as e:
            logger.error(
                f"LLM analysis failed: {str(e)}",
                extra={"error_type": type(e).__name__}
            )

            return {
                **state,
                "analysis": {
                    "summary": "Analysis unavailable due to error",
                    "main_topics": [],
                    "document_type": "unknown",
                    "key_insights": [],
                    "complexity_level": "unknown"
                },
                "analysis_success": False,
                "analysis_error": str(e)
            }
