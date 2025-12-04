"""Client for deployed A2A agents in DI environment."""

import logging
import httpx
import json
from typing import Dict, Any, Optional, List
from uuid import uuid4

logger = logging.getLogger(__name__)


class DeployedA2AClient:
    """Client for communicating with deployed A2A agents using JSON-RPC 2.0 protocol."""

    def __init__(self, agent_url: str, timeout: int = 300):
        """
        Initialize deployed A2A client.

        Args:
            agent_url: Base URL of the deployed agent
            timeout: Request timeout in seconds
        """
        self.agent_url = agent_url.rstrip('/') + '/'  # Ensure trailing slash for deployed agents
        self.timeout = timeout

    async def close(self):
        """Close the HTTP client - no-op since we create fresh clients."""
        pass

    async def send_request(
        self,
        task: str,
        data: Dict[str, Any],
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send request to deployed A2A agent using JSON-RPC 2.0 protocol.

        Args:
            task: Task type
            data: Task data
            context_id: Optional context ID for conversation

        Returns:
            Response dict with status and data
        """
        logger.info(f"Sending request to deployed agent: {self.agent_url}")
        
        timeout_config = httpx.Timeout(
            connect=30.0,
            read=self.timeout,
            write=30.0,
            pool=10.0
        )
        
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            try:
                # Format the message with task and data as JSON
                message_text = json.dumps({"task": task, "data": data})
                
                # Build JSON-RPC 2.0 request
                request_payload = {
                    "jsonrpc": "2.0",
                    "id": str(uuid4()),
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "message_id": f"msg_{task}_{uuid4().hex[:8]}",
                            "parts": [
                                {
                                    "kind": "text",
                                    "text": message_text
                                }
                            ],
                            "metadata": {
                                "stream": False
                            },
                            "context_id": context_id or f"{task}_thread_{uuid4().hex[:8]}"
                        }
                    }
                }
                
                logger.debug(f"Sending JSON-RPC request to {self.agent_url}")
                logger.debug(f"Request payload: {request_payload}")
                response = await client.post(
                    self.agent_url,
                    json=request_payload,
                    headers={"Content-Type": "application/json"}
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                logger.info(f"Received response from deployed agent")
                
                # Parse JSON-RPC 2.0 response
                if "result" not in response_data:
                    raise Exception(f"Invalid JSON-RPC response: {response_data}")
                
                result = response_data["result"]
                
                # Extract data from artifacts
                if "artifacts" in result and result["artifacts"]:
                    artifact = result["artifacts"][0]
                    if "parts" in artifact and artifact["parts"]:
                        part = artifact["parts"][0]
                        if "text" in part:
                            result_text = part["text"]
                            try:
                                parsed_result = json.loads(result_text)
                                logger.info(f"Successfully parsed agent response")
                                return parsed_result
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse result JSON: {e}")
                                return {
                                    "status": "error",
                                    "message": "Failed to parse agent response",
                                    "data": {"raw_response": result_text}
                                }
                
                # Fallback: return raw result if no artifacts
                logger.warning("No artifacts found in response, returning raw result")
                return {
                    "status": "completed",
                    "message": "Response received but no artifacts found",
                    "data": result
                }
                
            except httpx.TimeoutException:
                error_msg = f"Timeout after {self.timeout} seconds waiting for response from {self.agent_url}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "message": "Request timeout",
                    "data": {}
                }
            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP error {e.response.status_code} from {self.agent_url}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "message": f"HTTP error: {e.response.status_code}",
                    "data": {}
                }
            except Exception as e:
                error_msg = f"Communication error with {self.agent_url}: {str(e)}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "message": f"Communication error: {str(e)}",
                    "data": {}
                }

    async def health_check(self) -> bool:
        """
        Check health of deployed agent.

        Returns:
            True if agent is healthy, False otherwise
        """
        try:
            timeout_config = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                # Try health endpoint first
                health_url = f"{self.agent_url}/health"
                response = await client.get(health_url)
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Health check failed for {self.agent_url}: {e}")
            return False


class DeployedPolicyAnalysisClient(DeployedA2AClient):
    """Client for deployed Policy Analysis Agent."""

    def __init__(self, agent_url: str):
        super().__init__(agent_url)
        logger.info(f"Initialized DeployedPolicyAnalysisClient for {agent_url}")

    async def analyze_policy(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze policy and generate questionnaire.
        
        Args:
            policy_data: Dict containing policy_name, policy_text, page_references
            
        Returns:
            Dict with analysis results
        """
        policy_name = policy_data.get("policy_name", "Unknown Policy")
        logger.info(f"Analyzing policy '{policy_name}' using deployed agent")
        
        return await self.send_request(
            task="analyze_policy",
            data=policy_data
        )

    async def consolidate_questionnaire(self, consolidation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate multiple questionnaires.
        
        Args:
            consolidation_data: Dict containing question_chunks, policy_name, target_question_count
            
        Returns:
            Dict with consolidated questionnaire
        """
        policy_name = consolidation_data.get("policy_name", "Unknown Policy")
        logger.info(f"Consolidating questionnaire for '{policy_name}' using deployed agent")
        
        return await self.send_request(
            task="consolidate_questionnaire",
            data=consolidation_data
        )


class DeployedApplicationProcessingClient(DeployedA2AClient):
    """Client for deployed Application Processing Agent."""

    def __init__(self, agent_url: str):
        super().__init__(agent_url)
        logger.info(f"Initialized DeployedApplicationProcessingClient for {agent_url}")

    async def process_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process application and answer questionnaire.
        
        Args:
            application_data: Dict containing pages_data and questions
            
        Returns:
            Dict with processing results
        """
        pages_count = len(application_data.get("pages_data", []))
        questions_count = len(application_data.get("questions", []))
        logger.info(f"Processing application with {pages_count} pages and {questions_count} questions using deployed agent")
        
        return await self.send_request(
            task="process_application",
            data=application_data
        )


class DeployedDecisionMakingClient(DeployedA2AClient):
    """Client for deployed Decision Making Agent."""

    def __init__(self, agent_url: str):
        super().__init__(agent_url)
        logger.info(f"Initialized DeployedDecisionMakingClient for {agent_url}")

    async def evaluate_answers(
        self,
        answer_batch: List[Dict],
        policy_name: str,
        batch_number: int = 1
    ) -> Dict[str, Any]:
        """
        Evaluate answers against policy requirements.
        
        Args:
            answer_batch: List of answers to evaluate
            policy_name: Name of the policy
            batch_number: Batch identifier
            
        Returns:
            Dict with evaluation results
        """
        logger.info(f"Evaluating {len(answer_batch)} answers (batch {batch_number}) using deployed agent")
        
        evaluation_data = {
            "answer_batch": answer_batch,
            "policy_name": policy_name,
            "batch_number": batch_number
        }
        
        return await self.send_request(
            task="evaluate_answers",
            data=evaluation_data
        )

    async def generate_recommendation(
        self,
        policy_name: str,
        total_questions: int,
        requirements_met: int,
        requirements_not_met: int,
        unclear_requirements: int,
        missing_information: int,
        overall_risk_level: str,
        critical_issues: List[str] = None,
        found_answers: List[Dict] = None,
        patient_context: Dict[str, Any] = None,
        reference_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate final authorization recommendation.
        
        Args:
            policy_name: Policy name
            total_questions: Total number of questions
            requirements_met: Count of met requirements
            requirements_not_met: Count of unmet requirements
            unclear_requirements: Count of unclear requirements
            missing_information: Count of missing information
            overall_risk_level: Overall risk assessment
            critical_issues: List of critical findings
            found_answers: List of found answers with clinical details
            patient_context: Patient details for decision context
            reference_context: Reference data (dates, system info)
            
        Returns:
            Dict with recommendation
        """
        logger.info(f"Generating recommendation for '{policy_name}' using deployed agent")
        
        if critical_issues is None:
            critical_issues = []
        if found_answers is None:
            found_answers = []
        if patient_context is None:
            patient_context = {}
        if reference_context is None:
            reference_context = {}
        
        recommendation_data = {
            "policy_name": policy_name,
            "total_questions": total_questions,
            "requirements_met": requirements_met,
            "requirements_not_met": requirements_not_met,
            "unclear_requirements": unclear_requirements,
            "missing_information": missing_information,
            "overall_risk_level": overall_risk_level,
            "critical_issues": critical_issues,
            "found_answers": found_answers,
            "patient_context": patient_context,
            "reference_context": reference_context
        }
        
        return await self.send_request(
            task="generate_recommendation",
            data=recommendation_data
        )