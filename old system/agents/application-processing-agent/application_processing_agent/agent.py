"""Policy Analysis Agent for processing medical policy documents."""

import logging
import json
import re
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

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

class ApplicationProcessingAgent:
    """
    Simplified Application Processing Agent.
    
    This agent has ONE primary function: process_application
    - Extracts patient summary from application documents
    - Answers questionnaire questions using multimodal analysis
    - Returns complete application analysis in a single operation
    
    Image preparation and database operations are handled by the backend.
    """

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'application/json']

    def __init__(self):
        """Initialize the agent with multimodal LLM."""
        logger.info("Initializing ApplicationProcessingAgent")
        try:
            self.model = get_llm()
            self.cost_tracker = CostTracker(
                pricing=agent_settings.AP_LLM_PRICING,
                model=agent_settings.AP_LLM_MODEL
            )
            logger.info("ApplicationProcessingAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ApplicationProcessingAgent: {e}", exc_info=True)
            raise RuntimeError(f"Agent initialization failed: {e}") from e

    def _extract_json_from_response(self, response_text: str) -> dict:
        """Extract JSON from LLM response."""
        if not response_text.strip():
            return {}

        try:
            parsed = json.loads(response_text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

        json_pattern = r'\{[\s\S]*\}'
        matches = re.findall(json_pattern, response_text)

        for match in matches:
            try:
                parsed = json.loads(match)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                continue

        logger.warning(f"Could not extract valid JSON from response: {response_text[:200]}...")
        return {}

    async def process_application(self, application_data: dict) -> dict[str, Any]:
        """
        Complete application processing: extract patient summary AND answer questionnaire questions.
        
        Args:
            application_data: Dict containing pages_data and questions
        
        Returns:
            Dict with patient summary and extracted answers
        """
        try:
            
            pages_data = application_data.get("pages_data", [])
            questions = application_data.get("questions", [])
            
            logger.info(f"Processing application with {len(pages_data)} pages and {len(questions)} questions")
            
            if not pages_data:
                return {
                    "status": "error",
                    "message": "No application pages provided",
                    "data": {}
                }

            if not questions:
                return {
                    "status": "error", 
                    "message": "No questions provided",
                    "data": {}
                }

            # Process all pages together
            return await self._process_application_data(pages_data, questions)

        except Exception as e:
            logger.error(f"Error during application processing: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Application processing error: {str(e)}",
                "data": {}
            }



    async def _process_application_data(self, pages_data: list, questions: list) -> dict[str, Any]:
        """
        Process complete application data with all pages and questions.
        
        Args:
            pages_data: List of all page data (pre-processed by backend)
            questions: List of all questions to answer
            
        Returns:
            Dict with comprehensive patient summary and extracted answers
        """
        try:
            content = []
            
            content.append({
                "type": "text",
                "text": f"""TASK: Analyze this PRIOR AUTHORIZATION medical application to:
1. Extract comprehensive patient summary including all medical details
2. Answer all policy-based questionnaire questions with medical accuracy and specificity

CRITICAL MEDICAL DATA EXTRACTION FOCUS:
- Patient demographics: Full name, date of birth, patient ID, contact information
- Primary diagnosis: Exact medical condition with ICD-10/CPT codes 
- Secondary diagnoses: All comorbidities with dates of diagnosis and relevant codes
- Clinical measurements: Exact values with units (BMI, weight, height, vital signs, lab values)
- Physician information: Requesting physician name, specialty, contact information, credentials
- Insurance details: Carrier name, plan type, member ID, group number, effective dates
- Medical history: Previous procedures, treatments, hospitalizations with dates and outcomes
- Current medications: Complete list with exact dosages, frequencies, and administration routes
- Specialist evaluations: All consultations, evaluations, clearances with dates and recommendations
- Treatment timeline: Duration of conservative/prior treatments, failed interventions, progression
- Procedure details: Specific procedures, treatments, or interventions requested
- Clinical documentation: Test results, imaging, assessments, progress notes

BE THOROUGH AND POLICY-AGNOSTIC: Extract all medical information regardless of the specific condition or procedure type.

QUESTIONS TO ANSWER (Based on Medical Policy Requirements):
"""
            })

            for idx, q in enumerate(questions):
                question_id = q.get('question_id', q.get('id', idx + 1))  # Use 1-based indexing
                content.append({
                    "type": "text",
                    "text": f"\nQ{idx + 1} (ID: {question_id}): {q['question_text']}\n"
                })

            content.append({
                "type": "text",
                "text": f"\nAPPLICATION PAGES (Pages {pages_data[0]['page_number']}-{pages_data[-1]['page_number']}):\n"
            })

            # Add all pages with both text and images
            for page_data in pages_data:
                content.append({
                    "type": "text",
                    "text": f"\n--- PAGE {page_data['page_number']} TEXT ---\n{page_data.get('text', page_data.get('text_content', ''))}\n"
                })

                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{page_data['image_base64']}",
                        "detail": "low"  # Use low detail to reduce token usage
                    }
                })

            # Get current date for age and duration calculations
            current_date = datetime.now()
            current_date_str = current_date.strftime("%Y-%m-%d")
            current_date_readable = current_date.strftime("%B %d, %Y")
            current_year = current_date.strftime("%Y")

            content.append({
                "type": "text", 
                "text": f"""
PRIOR AUTHORIZATION ANALYSIS INSTRUCTIONS:

1. COMPREHENSIVE PATIENT MEDICAL SUMMARY:
   - Patient demographics: Full name, date of birth, patient ID, insurance information
   - Primary diagnosis: Main medical condition requiring authorization (with ICD-10 codes)
   - Secondary diagnoses: Relevant comorbidities and related conditions
   - Clinical measurements: Current BMI, weight, height, vital signs, lab values
   - Medical history: Previous surgeries, treatments, hospitalizations with dates
   - Current medications: Complete list with dosages and frequency
   - Physician evaluations: Specialist consultations, recommendations, evaluations
   - Treatment timeline: Duration of conservative management, failed treatments
   - Symptom progression: How condition has evolved over time

2. POLICY-BASED QUESTION ANSWERING:
   - Carefully review both TEXT and VISUAL content from pages {pages_data[0]['page_number']}-{pages_data[-1]['page_number']}
   - Extract specific medical data points (BMI values, dates, medications, diagnoses)
   - Look for checkboxes, form fields, and handwritten information in images
   - Provide EXACT values when found (e.g., "BMI: 42.3 kg/m²" not "high BMI")
   - Use 'NOT_FOUND' only when information is definitively not present
   - For numeric questions: Extract exact numbers with units
   - For date questions: Use MM/DD/YYYY format
   - For age questions: If age is not directly stated but date of birth is available, calculate age from DOB
   - For yes/no questions: Look for explicit confirmations or clear medical evidence
   - CRITICAL: Use the exact question_id shown in parentheses (ID: X) for each answer
   - Include source page numbers and exact text/visual evidence for all findings
   - CONFIDENCE SCORING: 
     * 95%: Explicitly stated in document with exact values/confirmations
     * 90%: Clear evidence from multiple sources or calculated from available data
     * 85%: Strong evidence but may require interpretation
     * 80%: Reasonable inference from available clinical information
     * 70%: Indirect evidence or partial information
   - BE SPECIFIC: Provide exact medical details rather than vague descriptions
   - CLINICAL CONTEXT: Always provide medical significance and clinical relevance of findings

SPECIAL INSTRUCTIONS FOR DATE AND AGE CALCULATIONS:
- IMPORTANT: Today's date is {current_date_str} ({current_date_readable}). Use this for all calculations.
- If a question asks for patient age and age is not directly stated, but you find a date of birth, calculate the current age
- For example: if DOB is 05/30/1982 and current date is {current_year}, the patient would be approximately {int(current_year) - 1982} years old
- Always include your calculation reasoning in the response and show your work
- Consider the specific month and day for precise age calculation

- DURATION/TIMELINE CALCULATIONS:
  * For "how long has patient had condition": Calculate from diagnosis date to current date ({current_date_str})
  * For "duration of conservative treatment": Calculate from treatment start to end dates
  * Look for phrases like "diagnosed in", "since", "for X years", "X months of"
  * Convert all durations to consistent units (months or years as requested)
- NUMERIC VALUES: Extract exact numbers with units (BMI: 45.2 kg/m², Weight: 285 lbs, BP: 140/90 mmHg)
- MEDICAL CODES: Look for ICD-10, CPT, HCPCS codes in forms or physician notes
- CLINICAL ASSESSMENTS: Extract physician assessments, evaluations, and clinical impressions
- TREATMENT RESPONSES: Document outcomes of previous treatments, effectiveness, complications
- FUNCTIONAL STATUS: Note any functional limitations, disability assessments, quality of life measures

CRITICAL CONSISTENCY CHECK: After answering all questions, cross-reference your answers to extract structured patient information. 
- If you found BMI in your answers, it MUST appear in patient_info.bmi
- If you found weight in your answers, it MUST appear in patient_info.weight  
- If you found any demographic or clinical data in answers, ensure it appears in the appropriate patient_info fields
- Ensure clinical_details sections reflect all relevant information found in your answers

OUTPUT FORMAT (JSON):
{{
  "patient_summary": "Comprehensive medical summary including all clinical details, diagnoses, treatments, and timeline...",
  "patient_info": {{
    "patient_name": "Full name or 'Not Found'",
    "patient_dob": "MM/DD/YYYY or 'Not Found'", 
    "patient_id": "Patient ID/Member ID or 'Not Found'",
    "insurance_info": "Insurance carrier and policy information or 'Not Found'",
    "primary_diagnosis": "Main diagnosis with ICD-10 code if available",
    "bmi": "MUST match BMI value from answers if found, otherwise 'Not Found'",
    "weight": "MUST match weight value from answers if found, otherwise 'Not Found'",
    "requesting_physician": "Physician name and specialty or 'Not Found'"
  }},
  "clinical_details": {{
    "comorbidities": ["List of secondary diagnoses/conditions"],
    "current_medications": ["List of medications with dosages"],
    "previous_treatments": ["List of failed conservative treatments with dates"],
    "lab_results": ["Recent lab values and test results"],
    "specialist_evaluations": ["Specialist consultations and recommendations"]
  }},
  "answers": [
    {{
      "question_id": "question ID exactly as provided",
      "question_text": "original question text",
      "answer_text": "extracted answer with specific values/units or 'NOT_FOUND'",
      "confidence_score": 95,
      "source_page_number": page_number_or_null,
      "source_text_snippet": "exact text from document or visual description",
      "source_type": "text|visual|both",
      "reasoning": "detailed explanation of how answer was extracted, including clinical significance and medical rationale for the finding"
    }}
  ]
}}

Analyze the application pages {pages_data[0]['page_number']}-{pages_data[-1]['page_number']}:
"""
            })

            message = HumanMessage(content=content)
            
            logger.info(f"Sending application processing request to LLM for pages {pages_data[0]['page_number']}-{pages_data[-1]['page_number']}")
            response = await self.model.ainvoke([message])
            
            # Calculate cost for this LLM call
            cost_data = self.cost_tracker.calculate_cost(response)
            
            # Parse the comprehensive response
            result_data = self._extract_json_from_response(response.content)
            
            if not result_data:
                logger.error("Invalid response format from LLM")
                return {
                    "status": "error",
                    "message": "Failed to parse LLM response",
                    "data": {}
                }
            
            patient_summary = result_data.get("patient_summary", "")
            patient_info = result_data.get("patient_info", {})
            answers = result_data.get("answers", [])
            
            # Helper function to safely get answer text as string
            def get_answer_text(answer):
                answer_text = answer.get('answer_text', '')
                if isinstance(answer_text, list):
                    # If it's a list, join the elements or take the first one
                    if answer_text:
                        return str(answer_text[0]) if len(answer_text) == 1 else ', '.join(str(item) for item in answer_text)
                    else:
                        return ''
                return str(answer_text)
            
            # Normalize answer_text to ensure it's always a string
            normalized_answers = []
            for answer in answers:
                normalized_answer = answer.copy()
                normalized_answer['answer_text'] = get_answer_text(answer)
                normalized_answers.append(normalized_answer)
            
            found_answers = [a for a in normalized_answers if a.get('answer_text', '').strip() != 'NOT_FOUND']
            not_found_answers = [a for a in normalized_answers if a.get('answer_text', '').strip() == 'NOT_FOUND']
            
            logger.info(f"Processing complete: {len(normalized_answers)} answers extracted ({len(found_answers)} found, {len(not_found_answers)} not found), summary length: {len(patient_summary)} chars")

            return {
                "status": "completed",
                "message": f"Successfully processed application: {len(found_answers)} answers found",
                "data": {
                    "patient_summary": patient_summary,
                    "patient_info": patient_info,
                    "answers": normalized_answers,
                    "summary": {
                        "pages_processed": len(pages_data),
                        "questions_answered": len(found_answers),
                        "questions_not_found": len(not_found_answers),
                        "summary_length": len(patient_summary)
                    },
                    "cost_info": {
                        "input_tokens": cost_data.input_tokens,
                        "output_tokens": cost_data.output_tokens,
                        "total_cost": cost_data.total_cost,
                        "model": cost_data.model
                    }
                }
            }

        except Exception as e:
            logger.error(f"Error during application processing: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Application processing error: {str(e)}",
                "data": {}
            }

    async def invoke(self, task: str, input_data: dict) -> dict[str, Any]:
        """
        Simplified invoke method - single primary task.

        Args:
            task: Task type ('process_application')
            input_data: Input data for the task

        Returns:
            Dict containing the response
        """
        logger.info(f"Invoking agent for task: {task}")

        try:
            if task == "process_application":
                return await self.process_application(input_data)
            else:
                return {
                    "status": "error",
                    "message": f"Unknown task type: {task}. Supported: process_application",
                    "data": {}
                }

        except Exception as e:
            logger.error(f"Error during invoke: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Agent error: {str(e)}",
                "data": {}
            }