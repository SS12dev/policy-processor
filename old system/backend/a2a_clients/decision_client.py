import logging
from typing import Dict, Any, List

try:
    from .base_client import A2ABaseClient
except ImportError:
    from base_client import A2ABaseClient

logger = logging.getLogger(__name__)

class DecisionMakingClient(A2ABaseClient):
    """Client for Decision Making Agent (Agent 3)"""

    def __init__(self, agent_url: str):
        super().__init__(agent_url)
        logger.info(f"Initialized DecisionMakingClient for {agent_url}")

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
        logger.info(f"Evaluating {len(answer_batch)} answers (batch {batch_number})")

        result = await self.send_request(
            task="evaluate_answers",
            data={
                "answer_batch": answer_batch,
                "policy_name": policy_name,
                "batch_number": batch_number
            }
        )

        if result.get("status") == "completed":
            evaluations = result.get("data", {}).get("evaluations", [])
            logger.info(f"Successfully evaluated {len(evaluations)} answers")
        else:
            logger.error(f"Failed to evaluate answers: {result.get('message', 'Unknown error')}")

        return result

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
        logger.info("Generating final authorization recommendation")

        if critical_issues is None:
            critical_issues = []
        if found_answers is None:
            found_answers = []
        if patient_context is None:
            patient_context = {}
        if reference_context is None:
            reference_context = {}

        result = await self.send_request(
            task="generate_recommendation",
            data={
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
        )

        if result.get("status") == "completed":
            recommendation = result.get("data", {}).get("recommendation", {})
            logger.info(f"Generated recommendation: {recommendation.get('recommendation', 'Unknown')}")
        else:
            logger.error(f"Failed to generate recommendation: {result.get('message', 'Unknown error')}")

        return result