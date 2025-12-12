"""
Response Formatter Node
Formats the final response for the client.
"""

import json
from typing import Dict, Any
from datetime import datetime

from settings import settings
from utils.logger import get_logger
from utils.metrics import track_node_execution

logger = get_logger(__name__)


async def format_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the final response with all extracted information.

    Args:
        state: Complete graph state with all processing results

    Returns:
        Updated state with formatted_response
    """
    with track_node_execution("response_formatter"):
        try:
            logger.info("Formatting response")

            # Build the response structure
            response = {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "agent": {
                    "name": settings.agent_name,
                    "version": settings.agent_version
                },
                "document_info": {
                    "page_count": state.get("page_count", 0),
                    "word_count": state.get("word_count", 0),
                    "size_mb": state.get("pdf_size_mb", 0),
                    "metadata": state.get("pdf_metadata", {})
                },
                "extraction_results": {
                    "headings": {
                        "count": state.get("heading_count", 0),
                        "items": [
                            {
                                "level": h.get("level"),
                                "text": h.get("text"),
                                "type": h.get("type"),
                                "line_number": h.get("line_number")
                            }
                            for h in state.get("headings", [])
                        ]
                    },
                    "keywords": {
                        "count": state.get("keyword_count", 0),
                        "items": state.get("keywords", [])
                    }
                },
                "analysis": state.get("analysis", {}),
                "processing_steps": {
                    "pdf_parsing": state.get("parsing_success", False),
                    "heading_extraction": state.get("heading_extraction_success", False),
                    "keyword_extraction": state.get("keyword_extraction_success", False),
                    "llm_analysis": state.get("analysis_success", False)
                }
            }

            # Add errors if any occurred
            errors = []
            if not state.get("parsing_success"):
                errors.append(f"PDF Parsing: {state.get('parsing_error', 'Unknown error')}")
            if not state.get("heading_extraction_success"):
                errors.append(f"Heading Extraction: {state.get('heading_extraction_error', 'Unknown error')}")
            if not state.get("keyword_extraction_success"):
                errors.append(f"Keyword Extraction: {state.get('keyword_extraction_error', 'Unknown error')}")
            if not state.get("analysis_success") and settings.enable_llm_analysis:
                errors.append(f"LLM Analysis: {state.get('analysis_error', 'Unknown error')}")

            if errors:
                response["warnings"] = errors

            # Create human-readable summary
            summary = create_text_summary(response)

            logger.info(
                "Response formatted successfully",
                extra={
                    "heading_count": response["extraction_results"]["headings"]["count"],
                    "keyword_count": response["extraction_results"]["keywords"]["count"],
                    "has_warnings": len(errors) > 0
                }
            )

            return {
                **state,
                "formatted_response": response,
                "text_summary": summary,
                "response_ready": True
            }

        except Exception as e:
            logger.error(
                f"Response formatting failed: {str(e)}",
                extra={"error_type": type(e).__name__}
            )

            # Create minimal error response
            error_response = {
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "agent": {
                    "name": settings.agent_name,
                    "version": settings.agent_version
                }
            }

            return {
                **state,
                "formatted_response": error_response,
                "text_summary": f"Error formatting response: {str(e)}",
                "response_ready": False
            }


def create_text_summary(response: Dict[str, Any]) -> str:
    """
    Create a human-readable text summary of the analysis.

    Args:
        response: Formatted response dictionary

    Returns:
        Text summary string
    """
    lines = []

    lines.append("=" * 80)
    lines.append("DOCUMENT ANALYSIS RESULTS")
    lines.append("=" * 80)
    lines.append("")

    # Document Info
    doc_info = response["document_info"]
    lines.append("Document Information:")
    lines.append(f"  - Pages: {doc_info['page_count']}")
    lines.append(f"  - Words: {doc_info['word_count']:,}")
    lines.append(f"  - Size: {doc_info['size_mb']:.2f} MB")

    metadata = doc_info.get("metadata", {})
    if metadata.get("title"):
        lines.append(f"  - Title: {metadata['title']}")
    if metadata.get("author"):
        lines.append(f"  - Author: {metadata['author']}")

    lines.append("")

    # Headings
    headings_data = response["extraction_results"]["headings"]
    lines.append(f"Extracted Headings ({headings_data['count']}):")
    for heading in headings_data["items"][:15]:  # Show first 15
        level_marker = "  " * (heading["level"] - 1)
        lines.append(f"{level_marker}- {heading['text']}")

    if headings_data['count'] > 15:
        lines.append(f"  ... and {headings_data['count'] - 15} more")

    lines.append("")

    # Keywords
    keywords_data = response["extraction_results"]["keywords"]
    lines.append(f"Keywords ({keywords_data['count']}):")
    keywords_text = ", ".join(keywords_data["items"][:20])
    lines.append(f"  {keywords_text}")

    if keywords_data['count'] > 20:
        lines.append(f"  ... and {keywords_data['count'] - 20} more")

    lines.append("")

    # LLM Analysis
    analysis = response.get("analysis", {})
    if analysis and analysis.get("summary"):
        lines.append("Document Analysis:")
        lines.append(f"  Summary: {analysis['summary']}")
        lines.append(f"  Type: {analysis.get('document_type', 'unknown')}")
        lines.append(f"  Complexity: {analysis.get('complexity_level', 'unknown')}")

        if analysis.get("main_topics"):
            lines.append(f"  Main Topics: {', '.join(analysis['main_topics'])}")

        if analysis.get("key_insights"):
            lines.append("  Key Insights:")
            for insight in analysis["key_insights"]:
                lines.append(f"    - {insight}")

    lines.append("")

    # Warnings
    if response.get("warnings"):
        lines.append("Warnings:")
        for warning in response["warnings"]:
            lines.append(f"  - {warning}")
        lines.append("")

    lines.append("=" * 80)
    lines.append(f"Analysis completed at {response['timestamp']}")
    lines.append("=" * 80)

    return "\n".join(lines)
