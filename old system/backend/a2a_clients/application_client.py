import logging
from typing import Dict, Any, List

try:
    from .base_client import A2ABaseClient
except ImportError:
    from base_client import A2ABaseClient

logger = logging.getLogger(__name__)

class ApplicationProcessingClient(A2ABaseClient):
    """Client for Application Processing Agent (Agent 2) - Simplified single-task approach"""

    def __init__(self, agent_url: str):
        super().__init__(agent_url)
        logger.info(f"Initialized ApplicationProcessingClient for {agent_url}")

    async def process_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete application processing - extract patient summary AND answer questions in one call.
        Backend handles optimal chunking, agent processes pre-chunked data.
        
        Args:
            application_data: Dict containing pages_data (pre-chunked by backend) and questions

        Returns:
            Dict with patient info, summary, and all answers
        """
        pages_data = application_data.get("pages_data", [])
        questions = application_data.get("questions", [])
        
        # Log chunk characteristics if available
        chunk_info = ""
        if pages_data:
            page_range = f"pages {pages_data[0].get('page_number', 1)}-{pages_data[-1].get('page_number', len(pages_data))}"
            total_chars = sum(len(page.get('text_content', page.get('text', ''))) for page in pages_data)
            chunk_info = f" ({page_range}, {total_chars} chars)"
        
        logger.info(f"APPLICATION_CLIENT_DEBUG: process_application ENTRY")
        logger.info(f"APPLICATION_CLIENT_DEBUG: Processing application chunk with {len(pages_data)} pages and {len(questions)} questions{chunk_info}")
        logger.info(f"APPLICATION_CLIENT_DEBUG: About to call send_request for process_application")

        result = await self.send_request(
            task="process_application",
            data=application_data
        )

        logger.info(f"APPLICATION_CLIENT_DEBUG: send_request returned: {result.get('status', 'unknown')}")

        if result.get("status") == "completed":
            app_data = result.get("data", {})
            answers = app_data.get("answers", [])
            found_answers = [a for a in answers if a.get('answer_text', '').strip() != 'NOT_FOUND']
            summary_length = len(app_data.get("patient_summary", ""))
            logger.info(f"APPLICATION_CLIENT_DEBUG: Found {len(found_answers)}/{len(answers)} valid answers, summary: {summary_length} chars")
            logger.info(f"APPLICATION_CLIENT_DEBUG: Patient info keys: {list(app_data.get('patient_info', {}).keys())}")
            logger.info(f"Successfully processed chunk: {len(found_answers)}/{len(answers)} answers found, {summary_length} char summary")
        else:
            logger.error(f"APPLICATION_CLIENT_DEBUG: Failed to process chunk: {result.get('message', 'Unknown error')}")
            logger.error(f"Failed to process chunk: {result.get('message', 'Unknown error')}")

        return result

    # Keep backward compatibility methods during transition
    async def extract_patient_info(self, pages_data: List[Dict]) -> Dict[str, Any]:
        """
        DEPRECATED: Use process_application() instead.
        Kept for backward compatibility during transition.
        """
        logger.warning("extract_patient_info() is deprecated. Use process_application() instead.")
        logger.info(f"Extracting patient info from {len(pages_data)} pages")

        result = await self.send_request(
            task="extract_patient_info",
            data={"pages_data": pages_data}
        )

        if result.get("status") == "completed":
            patient_info = result.get("data", {}).get("patient_info", {})
            logger.info(f"Successfully extracted patient info: {patient_info.get('patient_name', 'Unknown')}")
        else:
            logger.error(f"Failed to extract patient info: {result.get('message', 'Unknown error')}")

        return result

    async def update_summary(
        self,
        page_batch: List[Dict],
        current_summary: str
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Use process_application() instead.
        Kept for backward compatibility during transition.
        """
        logger.warning("update_summary() is deprecated. Use process_application() instead.")
        logger.info(f"Updating patient summary with {len(page_batch)} new pages")

        result = await self.send_request(
            task="update_summary",
            data={
                "page_batch": page_batch,
                "current_summary": current_summary
            }
        )

        if result.get("status") == "completed":
            summary = result.get("data", {}).get("summary", "")
            logger.info(f"Successfully updated summary ({len(summary)} chars)")
        else:
            logger.error(f"Failed to update summary: {result.get('message', 'Unknown error')}")

        return result

    async def extract_answers(
        self,
        page_batch: List[Dict],
        patient_summary: str,
        questions: List[Dict]
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Use process_application() instead.
        Kept for backward compatibility during transition.
        """
        logger.warning("extract_answers() is deprecated. Use process_application() instead.")
        logger.info(f"Extracting answers for {len(questions)} questions from {len(page_batch)} pages")

        result = await self.send_request(
            task="extract_answers",
            data={
                "page_batch": page_batch,
                "patient_summary": patient_summary,
                "questions": questions
            }
        )

        if result.get("status") == "completed":
            answers = result.get("data", {}).get("answers", [])
            logger.info(f"Successfully extracted {len(answers)} answers")
        else:
            logger.error(f"Failed to extract answers: {result.get('message', 'Unknown error')}")

        return result