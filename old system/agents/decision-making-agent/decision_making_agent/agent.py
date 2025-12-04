"""Decision Making Agent for evaluating answers and generating recommendations."""

import json
import logging
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

try:
    from .utils.llm import get_llm
    from .utils.cost_tracker import CostTracker
    from .utils.settings import agent_settings
except ImportError:
    from utils.llm import get_llm
    from utils.cost_tracker import CostTracker
    from utils.settings import agent_settings

logger = logging.getLogger(__name__)

class ResponseFormat(BaseModel):
    """Structured response format for the agent."""
    status: Literal['input_required', 'completed', 'error'] = 'completed'
    message: str
    data: dict = {}

class DecisionMakingAgent:
    """
    Decision Making Agent for evaluating answers and generating recommendations.
    This agent focuses ONLY on LLM tasks: answer evaluation and recommendation generation.
    Data aggregation and database operations are handled by the backend.
    """

    SYSTEM_INSTRUCTION = """You are a senior medical director and prior authorization specialist with expertise in clinical review and medical necessity determinations.

Your task is to evaluate prior authorization applications against medical policy requirements and generate evidence-based authorization decisions.

CLINICAL EXPERTISE AREAS:
- Medical necessity assessment for surgical and medical procedures
- Risk stratification based on patient clinical profiles
- Policy compliance evaluation and regulatory requirements
- Evidence-based medicine and clinical guidelines
- Healthcare quality and safety standards

KEY RESPONSIBILITIES:
- Assess clinical appropriateness of requested procedures/treatments
- Evaluate medical necessity based on established criteria
- Identify safety concerns and contraindications
- Assign risk levels based on clinical evidence and policy compliance
- Generate final authorization decisions (APPROVE/DENY/PEND) with detailed rationale
- Ensure decisions align with evidence-based medical standards
- Provide clear, defensible clinical reasoning for all determinations

DECISION CRITERIA:
- Medical necessity: Does the patient's condition warrant the requested treatment?
- Policy compliance: Are all required criteria met according to the medical policy?
- Safety assessment: Are there contraindications or unacceptable risks?
- Alternative treatments: Have appropriate conservative treatments been attempted?
- Documentation adequacy: Is sufficient clinical evidence provided?

Return structured JSON responses with detailed clinical reasoning."""

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'application/json']

    def __init__(self):
        """Initialize the agent with LLM."""
        logger.info("Initializing DecisionMakingAgent")
        try:
            self.model = get_llm()
            self.cost_tracker = CostTracker(
                pricing=agent_settings.DM_LLM_PRICING,
                model=agent_settings.DM_LLM_MODEL
            )
            # Removed self.call_costs to ensure thread safety for concurrent requests
            logger.info("DecisionMakingAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DecisionMakingAgent: {e}", exc_info=True)
            raise RuntimeError(f"Agent initialization failed: {e}") from e

    def extract_json_from_response(self, response_text: str) -> list:
        """Extract JSON array from LLM response."""
        if not response_text.strip():
            return []

        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

        json_pattern = r'\[[\s\S]*?\]'
        matches = re.findall(json_pattern, response_text)

        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                continue

        logger.warning(f"Could not extract valid JSON from response: {response_text[:200]}...")
        return []

    async def evaluate_answer_batch(
        self,
        answer_batch: list[dict],
        policy_name: str,
        batch_number: int,
        call_costs: list = None
    ) -> dict[str, Any]:
        """
        Evaluate a batch of answers against policy requirements.

        Args:
            answer_batch: List of answers to evaluate
            policy_name: Name of the policy
            batch_number: Batch identifier
            call_costs: List to track costs for this request (for concurrency safety)

        Returns:
            Dict with evaluation results
        """
        # Initialize cost tracking if not provided (for backward compatibility)
        if call_costs is None:
            call_costs = []
        logger.info(f"Evaluating batch {batch_number} with {len(answer_batch)} answers")

        questions_text = ""
        for i, answer in enumerate(answer_batch):
            question_num = answer.get('question_id', i + 1)
            answer_text = answer.get('answer_text', 'NOT_FOUND')
            confidence = answer.get('confidence_score', 0)
            source_page = answer.get('source_page_number', 'N/A')
            reasoning = answer.get('reasoning', '')
            
            # Indicate whether this was actually found or not
            found_status = "FOUND" if answer_text != "NOT_FOUND" and answer_text != "Not Found" else " NOT FOUND"
            
            questions_text += f"Q{question_num}: {answer.get('question_text', '')}\n"
            questions_text += f"Status: {found_status}\n"
            questions_text += f"Answer: {answer_text}\n"
            questions_text += f"Confidence: {confidence}%\n"
            questions_text += f"Source Page: {source_page}\n"
            if reasoning:
                questions_text += f"Agent Reasoning: {reasoning}\n"
            questions_text += "\n"

        # Import datetime to get current date
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""
TASK: For each question, determine if the answer meets the clinical requirement.

{questions_text}

EVALUATION LOGIC:
• If answer = "Yes" to a requirement question → meets_requirement = "Yes"
• If answer = "No" to a contraindication question → meets_requirement = "Yes" (no contraindications is good)
• If answer provides clinical data that satisfies the question → meets_requirement = "Yes"
• If answer = "NOT_FOUND" → meets_requirement = "Missing"
• Only mark "No" if there are documented contraindications or criteria clearly not met

APPROACH:
1. Read the question - what is it asking for?
2. Look at the answer - does it provide evidence that requirement is satisfied?
3. Mark "Yes" if clinical evidence supports meeting the requirement
4. Be generous with "Yes" - if answer supports approval, mark as "Yes"

CRITICAL: Return EXACTLY this JSON format:
[
  {{
    "question_id": "1",
    "question_text": "Original question text",
    "extracted_answer": "Answer from application",
    "meets_requirement": "Yes",
    "clinical_significance": "Important",
    "risk_assessment": "Low",
    "clinical_rationale": "Brief reason why this meets the requirement",
    "policy_impact": "Supports approval",
    "additional_info_needed": ""
  }}
]

MANDATORY: For each question with a "Yes" answer or positive clinical finding, you MUST set meets_requirement = "Yes"

Evaluate all questions:
"""

        try:
            message = HumanMessage(content=prompt)
            response = await self.model.ainvoke([message])
            
            logger.info(f"RAW LLM EVALUATION RESPONSE:\n{response.content}")
            
            # Track cost for this LLM call (using request-local list for concurrency safety)
            cost_data = self.cost_tracker.calculate_cost(response)
            call_costs.append(cost_data)

            evaluations = self.extract_json_from_response(response.content)
            
            logger.info(f"PARSED EVALUATIONS: {json.dumps(evaluations, indent=2) if evaluations else 'NONE'}")

            if evaluations:
                logger.info(f"Evaluated {len(evaluations)} questions in batch {batch_number}")
            else:
                logger.warning(f"No evaluations extracted from batch {batch_number}")

            # Get cost for this specific call (last one added, using request-local list)
            current_cost = call_costs[-1] if call_costs else None
            cost_info = {
                "input_tokens": current_cost.input_tokens if current_cost else 0,
                "output_tokens": current_cost.output_tokens if current_cost else 0,
                "total_cost": current_cost.total_cost if current_cost else 0.0,
                "model": current_cost.model if current_cost else self.cost_tracker.model
            }
            
            return {
                "status": "completed",
                "message": f"Evaluated {len(evaluations)} answers",
                "data": {
                    "evaluations": evaluations,
                    "cost_info": cost_info
                }
            }

        except Exception as e:
            logger.error(f"Failed to evaluate batch: {e}")
            return {
                "status": "error",
                "message": f"Failed to evaluate batch: {str(e)}",
                "data": {"evaluations": []}
            }

    async def generate_recommendation(
        self,
        policy_name: str,
        total_questions: int,
        requirements_met: int,
        requirements_not_met: int,
        unclear_requirements: int,
        missing_information: int,
        overall_risk_level: str,
        critical_issues: list[str],
        found_answers: list[dict] = None,
        patient_context: dict = None,
        reference_context: dict = None,
        call_costs: list = None
    ) -> dict[str, Any]:
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
            call_costs: List to track costs for this request (for concurrency safety)

        Returns:
            Dict with recommendation
        """
        logger.info("Generating final authorization recommendation")

        critical_issues_text = "\n".join(critical_issues) if critical_issues else "None"

        # Calculate percentages for decision making
        total_evaluated = requirements_met + requirements_not_met + unclear_requirements
        met_percentage = (requirements_met / total_questions * 100) if total_questions > 0 else 0
        
        # Format found answers for detailed clinical reasoning
        found_answers_text = "None documented"
        if found_answers:
            found_answers_lines = []
            for ans in found_answers[:10]:  # Limit to first 10 to avoid token overflow
                q_text = ans.get('question_text', '')[:100] + '...' if len(ans.get('question_text', '')) > 100 else ans.get('question_text', '')
                answer_text = ans.get('answer_text', '')
                meets_req = ans.get('meets_requirement', 'Unknown')
                found_answers_lines.append(f"• Q: {q_text}\n  A: {answer_text} (Assessment: {meets_req})")
            found_answers_text = "\n".join(found_answers_lines)
        
        # Get reference context with current date
        if reference_context is None:
            reference_context = {}
        
        current_date = reference_context.get('current_date')
        if not current_date:
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Format patient context information if available
        patient_info = ""
        if patient_context:
            patient_details = []
            for key, value in patient_context.items():
                if value and str(value).strip():
                    patient_details.append(f"- {key.replace('_', ' ').title()}: {value}")
            if patient_details:
                patient_info = f"\nPATIENT CONTEXT:\n" + "\n".join(patient_details) + "\n"
        
        # Format reference context information if available
        reference_info = ""
        if reference_context and len(reference_context) > 1:  # More than just current_date
            ref_details = []
            for key, value in reference_context.items():
                if key != 'current_date' and value and str(value).strip():
                    ref_details.append(f"- {key.replace('_', ' ').title()}: {value}")
            if ref_details:
                reference_info = f"\nREFERENCE CONTEXT:\n" + "\n".join(ref_details) + "\n"
        
        prompt = f"""
PRIOR AUTHORIZATION DECISION for {policy_name}

EVALUATION RESULTS:
✅ CRITERIA MET: {requirements_met} out of {total_questions} total ({met_percentage:.1f}%)
❌ CRITERIA NOT MET: {requirements_not_met}
📋 MISSING DOCS: {missing_information}
❓ UNCLEAR: {unclear_requirements}

CLINICAL EVIDENCE FOUND:
{found_answers_text}

DECISION LOGIC:
📋 APPROVE if: Met ≥ {int(total_questions * 0.75)} ({int(total_questions * 0.75)}/{total_questions}) AND Not Met ≤ 2
📋 Current: Met = {requirements_met}, Not Met = {requirements_not_met}

APPROVAL CHECK:
- Does {requirements_met} ≥ {int(total_questions * 0.75)}? {requirements_met >= int(total_questions * 0.75)}
- Does {requirements_not_met} ≤ 2? {requirements_not_met <= 2}
- If both true → APPROVE

CLINICAL OVERRIDE: If clinical findings show strong evidence supporting medical necessity and policy compliance → APPROVE

Based on the numbers above: Met={requirements_met}, Not Met={requirements_not_met}, Missing={missing_information}
Review the clinical evidence above and make your decision.

OUTPUT FORMAT (JSON):
```json
{{
    "recommendation": "APPROVE|PEND|DENY",
    "confidence_score": 0.85,
    "primary_clinical_rationale": "Primary medical reason for this authorization decision",
    "medical_necessity_assessment": "Detailed clinical evaluation of medical necessity",
    "detailed_reasoning": "Comprehensive explanation of decision factors, policy criteria analysis, clinical evidence review, and step-by-step rationale leading to this decision",
    "decision_factors": {{
        "criteria_met": ["List specific policy criteria that were satisfied"],
        "criteria_not_met": ["List specific policy criteria that were not satisfied"],
        "clinical_strengths": ["Clinical factors supporting approval"],
        "clinical_concerns": ["Clinical factors raising concerns"],
        "safety_considerations": ["Patient safety factors considered"],
        "policy_compliance": ["Policy requirements and compliance status"]
    }},
    "supporting_clinical_factors": ["Clinical factor 1", "Clinical factor 2", "Clinical factor 3"],
    "clinical_concerns": ["Medical concern 1", "Safety concern 2"],
    "required_actions": ["Specific action 1", "Clinical documentation needed 2"],
    "review_priority": "Routine|Urgent|Expedited",
    "clinical_notes": "Additional clinical considerations or recommendations"
}}
```

Render final clinical authorization determination:
"""

        try:
            # Initialize cost tracking if not provided (for backward compatibility)
            if call_costs is None:
                call_costs = []
                
            message = HumanMessage(content=prompt)
            response = await self.model.ainvoke([message])
            
            # Track cost for this LLM call (using request-local list for concurrency safety)
            cost_data = self.cost_tracker.calculate_cost(response)
            call_costs.append(cost_data)

            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                recommendation = json.loads(json_match.group(0))
            else:
                recommendation = {}

            logger.info(f"Generated recommendation: {recommendation.get('recommendation', 'Unknown')}")

            # Get cost for this specific call (last one added, using request-local list)
            current_cost = call_costs[-1] if call_costs else None
            cost_info = {
                "input_tokens": current_cost.input_tokens if current_cost else 0,
                "output_tokens": current_cost.output_tokens if current_cost else 0,
                "total_cost": current_cost.total_cost if current_cost else 0.0,
                "model": current_cost.model if current_cost else self.cost_tracker.model
            }

            return {
                "status": "completed",
                "message": "Recommendation generated",
                "data": {
                    "recommendation": recommendation,
                    "cost_info": cost_info
                }
            }

        except Exception as e:
            logger.error(f"Failed to generate recommendation: {e}")
            return {
                "status": "error",
                "message": f"Failed to generate recommendation: {str(e)}",
                "data": {}
            }

    async def invoke(self, task: str, input_data: dict) -> dict[str, Any]:
        """
        Invoke agent for specific task.

        Args:
            task: Task type ('evaluate_answers' or 'generate_recommendation')
            input_data: Input data for the task

        Returns:
            Dict containing the response
        """
        logger.info(f"Invoking agent for task: {task}")
        
        # Initialize cost tracking for this specific request (ensures concurrency safety)
        call_costs = []

        try:
            if task == "evaluate_answers":
                answer_batch = input_data.get("answer_batch", [])
                policy_name = input_data.get("policy_name", "Unknown Policy")
                batch_number = input_data.get("batch_number", 1)

                if not answer_batch:
                    return {
                        "status": "error",
                        "message": "No answers provided for evaluation",
                        "data": {}
                    }

                return await self.evaluate_answer_batch(answer_batch, policy_name, batch_number, call_costs)

            elif task == "generate_recommendation":
                required_fields = [
                    "policy_name", "total_questions", "requirements_met",
                    "requirements_not_met", "unclear_requirements",
                    "missing_information", "overall_risk_level"
                ]

                for field in required_fields:
                    if field not in input_data:
                        return {
                            "status": "error",
                            "message": f"Missing required field: {field}",
                            "data": {}
                        }

                return await self.generate_recommendation(
                    policy_name=input_data["policy_name"],
                    total_questions=input_data["total_questions"],
                    requirements_met=input_data["requirements_met"],
                    requirements_not_met=input_data["requirements_not_met"],
                    unclear_requirements=input_data["unclear_requirements"],
                    missing_information=input_data["missing_information"],
                    overall_risk_level=input_data["overall_risk_level"],
                    critical_issues=input_data.get("critical_issues", []),
                    found_answers=input_data.get("found_answers", []),
                    patient_context=input_data.get("patient_context", {}),
                    reference_context=input_data.get("reference_context", {}),
                    call_costs=call_costs
                )

            else:
                return {
                    "status": "error",
                    "message": f"Unknown task type: {task}",
                    "data": {}
                }

        except Exception as e:
            logger.error(f"Error during invoke: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Agent error: {str(e)}",
                "data": {}
            }