import logging
from typing import Dict, Any

try:
    from .base_client import A2ABaseClient
except ImportError:
    from base_client import A2ABaseClient

logger = logging.getLogger(__name__)

class PolicyAnalysisClient(A2ABaseClient):
    """Client for Policy Analysis Agent (Agent 1) - Simplified single-task approach"""

    def __init__(self, agent_url: str):
        super().__init__(agent_url)
        logger.info(f"Initialized PolicyAnalysisClient for {agent_url}")

    async def analyze_policy(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete policy analysis - extract criteria AND generate questionnaire in one call.
        
        Args:
            policy_data: Dict containing policy_name, policy_text, page_references, etc.

        Returns:
            Dict with both extracted criteria and generated questions
        """
        print(f"POLICY_CLIENT_DEBUG: analyze_policy ENTRY")
        policy_name = policy_data.get("policy_name", "Unknown Policy")
        policy_text = policy_data.get("policy_text", "")
        print(f"POLICY_CLIENT_DEBUG: Processing {policy_name}, text length: {len(policy_text)}")
        logger.info(f"Analyzing complete policy '{policy_name}' with {len(policy_text)} characters")

        print(f"POLICY_CLIENT_DEBUG: About to call send_request for {policy_name}")
        try:
            result = await self.send_request(
                task="analyze_policy",  
                data=policy_data
            )
            print(f"POLICY_CLIENT_DEBUG: send_request SUCCESS, status: {result.get('status', 'NO_STATUS')}")
        except Exception as e:
            print(f"POLICY_CLIENT_DEBUG: send_request EXCEPTION: {e}")
            raise

        if result.get("status") == "completed":
            analysis_data = result.get("data", {})
            criteria = analysis_data.get("criteria", [])
            questions = analysis_data.get("questions", [])
            
            # Debug: Check if cost info is available
            print(f"POLICY_CLIENT_DEBUG: analysis_data keys: {list(analysis_data.keys())}")
            if "cost_info" in analysis_data:
                print(f"POLICY_CLIENT_DEBUG: Found cost_info: {analysis_data['cost_info']}")
            
            logger.info(f"Successfully analyzed policy: {len(criteria)} criteria, {len(questions)} questions")
        else:
            logger.error(f"Failed to analyze policy: {result.get('message', 'Unknown error')}")

        return result

    async def consolidate_questionnaire(self, consolidation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate multiple question chunks into optimized final questionnaire.
        
        Args:
            consolidation_data: Dict containing question_chunks, policy_name, target_question_count
            
        Returns:
            Dict with consolidated questionnaire
        """
        policy_name = consolidation_data.get("policy_name", "Unknown Policy")
        question_chunks = consolidation_data.get("question_chunks", [])
        
        logger.info(f"Consolidating questionnaire for '{policy_name}' from {len(question_chunks)} chunks")
        
        result = await self.send_request(
            task="consolidate_questionnaire",
            data=consolidation_data
        )
        
        if result.get("status") == "completed":
            consolidated_data = result.get("data", {})
            questionnaire = consolidated_data.get("consolidated_questionnaire", {})
            questions = questionnaire.get("questions", [])
            logger.info(f"Successfully consolidated questionnaire: {len(questions)} final questions")
        else:
            logger.error(f"Failed to consolidate questionnaire: {result.get('message', 'Unknown error')}")
        
        return result

    # Keep backward compatibility methods during transition
    async def extract_criteria(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        DEPRECATED: Use analyze_policy() instead. 
        Kept for backward compatibility during transition.
        """
        logger.warning("extract_criteria() is deprecated. Use analyze_policy() instead.")
        
        policy_name = data.get("policy_name", "Unknown Policy")
        chunks = data.get("chunks", [])
        logger.info(f"Extracting criteria from policy '{policy_name}' with {len(chunks)} chunks")

        result = await self.send_request(
            task="extract_policy_criteria",
            data=data
        )

        if result.get("status") == "completed":
            criteria = result.get("data", {}).get("criteria", [])
            logger.info(f"Successfully extracted {len(criteria)} criteria")
        else:
            logger.error(f"Failed to extract criteria: {result.get('message', 'Unknown error')}")

        return result

    async def generate_questions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        DEPRECATED: Use analyze_policy() instead.
        Kept for backward compatibility during transition.
        """
        logger.warning("generate_questions() is deprecated. Use analyze_policy() instead.")
        
        policy_name = data.get("policy_name", "Unknown Policy")
        criteria = data.get("criteria", [])
        logger.info(f"Generating questions for policy '{policy_name}' from {len(criteria)} criteria")

        result = await self.send_request(
            task="generate_questionnaire",
            data=data
        )

        if result.get("status") == "completed":
            questions = result.get("data", {}).get("questions", [])
            logger.info(f"Successfully generated {len(questions)} questions")
        else:
            logger.error(f"Failed to generate questions: {result.get('message', 'Unknown error')}")

        return result