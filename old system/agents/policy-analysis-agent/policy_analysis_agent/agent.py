"""Policy Analysis Agent for processing medical policy documents."""

import gc
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


class PolicyAnalysisAgent:
    """Policy Analysis Agent for processing medical policy documents.
    
    This agent has one primary function: analyze_policy
    - Extracts criteria from policy documents
    - Generates questionnaire questions from the criteria
    - Returns complete analysis in a single operation
    
    PDF processing and database operations are handled by the backend.
    """

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'application/json']

    def __init__(self):
        """Initialize the agent with LLM."""
        logger.info("Initializing PolicyAnalysisAgent")
        try:
            self.model = get_llm()
            self.cost_tracker = CostTracker(
                pricing=agent_settings.PA_LLM_PRICING,
                model=agent_settings.PA_LLM_MODEL
            )
            # Removed self.call_costs to ensure thread safety for concurrent requests
            logger.info("PolicyAnalysisAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PolicyAnalysisAgent: {e}", exc_info=True)
            raise RuntimeError(f"Agent initialization failed: {e}") from e

    async def _filter_medical_content(self, policy_text: str, call_costs: list) -> str:
        """
        Use LLM to intelligently filter policy text to keep only medical/clinical content
        and remove references, bibliography, revision history, and administrative sections.
        
        Args:
            policy_text: Raw policy document text
            call_costs: List to track costs for this request (for concurrency safety)
            
        Returns:
            Filtered text containing only medical/clinical content
        """
        if not policy_text.strip():
            return policy_text
        
        # If text is too short, likely doesn't need filtering
        if len(policy_text) < 1000:
            return policy_text
        
        # First, use simple heuristics to identify likely non-medical sections
        likely_admin_section = self._detect_administrative_section(policy_text)
        
        # Simple optimization: Skip LLM filtering if no administrative content detected
        if not likely_admin_section:
            logger.debug("No administrative sections detected, skipping LLM filtering")
            return policy_text
        
        logger.info(f"Administrative sections detected: {', '.join(likely_admin_section)}. Using LLM to filter content...")
        
        # Add context about detected administrative sections
        admin_context = ""
        if likely_admin_section:
            admin_context = f"\nNOTE: This document appears to contain administrative sections like: {', '.join(likely_admin_section)}. Be extra careful to remove these."
            
        filtering_prompt = f"""You are a medical content analyst. Your task is to filter policy document content to keep ONLY medical and clinical information relevant for prior authorization decisions.{admin_context}

KEEP these types of content:
- Medical necessity criteria and clinical indications
- Eligibility requirements (age, BMI, diagnosis criteria)
- Required medical procedures and evaluations
- Documentation requirements for medical records
- Diagnostic codes (ICD-10, CPT codes)
- Medical contraindications and exclusions
- Procedural specifications and facility requirements
- Clinical assessment requirements
- Treatment failure criteria and timelines

REMOVE these types of content:
- References and bibliography sections
- Academic citations and journal references
- Website links and external resources
- Revision history and document updates
- Contact information and administrative details
- Index and table of contents
- Legal disclaimers and copyright notices
- Navigation elements and formatting instructions

INSTRUCTIONS:
1. Read the content carefully
2. Extract and return ONLY the medical/clinical portions
3. Maintain the original formatting and structure of medical content
4. If a section mixes medical content with references, keep only the medical parts
5. If unsure whether content is medical, err on the side of keeping it
6. Return "REMOVE_SECTION" if the entire section should be filtered out
7. Return the filtered content exactly as it should appear

Content to analyze:
"""

        messages = [
            SystemMessage(content=filtering_prompt),
            HumanMessage(content=policy_text)
        ]
        
        try:
            response = await self.model.ainvoke(messages)
            
            # Track cost for this LLM call (using request-local list for concurrency safety)
            cost_data = self.cost_tracker.calculate_cost(response)
            call_costs.append(cost_data)
            
            filtered_text = response.content.strip()
            
            # If LLM says to remove the entire section, return empty string
            if filtered_text.upper() == "REMOVE_SECTION":
                logger.info("Content marked for complete removal by LLM")
                # MEMORY CLEANUP: Clear filtering variables  
                try:
                    del filtering_prompt, messages, response
                    gc.collect()
                except:
                    pass
                return ""
            
        except Exception as e:
            logger.warning(f"Error during LLM content filtering: {e}. Keeping original content.")
            filtered_text = policy_text
        
        # Calculate overall filtering statistics
        original_length = len(policy_text)
        filtered_length = len(filtered_text)
        
        if filtered_length < original_length:
            reduction_percent = ((original_length - filtered_length) / original_length) * 100
            logger.info(f"LLM content filtering: {filtered_length}/{original_length} characters remaining ({reduction_percent:.1f}% reduction)")
        else:
            logger.info("LLM content filtering: No significant content removed")
        
        # MEMORY CLEANUP: Clear filtering variables
        try:
            del filtering_prompt, messages
            if 'response' in locals():
                del response
            gc.collect()
        except:
            pass
        
        return filtered_text

    def _detect_administrative_section(self, text: str) -> list[str]:
        """
        Use simple heuristics to detect likely administrative sections.
        This provides guidance to the LLM but doesn't make filtering decisions.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected administrative section types
        """
        text_lower = text.lower()
        detected_sections = []
        
        # Simple keyword-based detection (guidance only)
        section_indicators = {
            'references': ['references', 'bibliography', 'citations'],
            'revision_history': ['revised', 'mptac review', 'updated', 'change log'],
            'websites': ['websites for additional information', 'available at:', 'accessed on'],
            'index': ['index', 'table of contents'],
            'contact': ['contact information', 'customer service', 'member services'],
            'legal': ['disclaimer', 'copyright', 'legal notice']
        }
        
        for section_type, keywords in section_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_sections.append(section_type)
        
        # Count potential citation patterns (guidance only)
        citation_patterns = [
            r'\(\d{4}\)',  # (2020)
            r'et\s+al\.',  # et al.
            r'https?://',  # URLs
            r'www\.',      # web addresses
        ]
        
        citation_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in citation_patterns)
        if citation_count > 10:  # High number of potential citations
            if 'references' not in detected_sections:
                detected_sections.append('references')
        
        return detected_sections

    def _extract_json_from_response(self, response_text: str) -> list:
        """Extract JSON from LLM response, handling markdown code blocks.
        
        Args:
            response_text: The raw response text from the LLM
            
        Returns:
            List of parsed JSON objects
        """
        if not response_text.strip():
            return []

        # Try to parse the response directly
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code blocks
        markdown_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
        markdown_matches = re.findall(markdown_pattern, response_text, re.IGNORECASE)
        
        for match in markdown_matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    return [parsed]
                elif isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                continue

        # Try to extract JSON objects with braces
        brace_pattern = r'\{[\s\S]*?\}'
        brace_matches = re.findall(brace_pattern, response_text)
        
        for match in brace_matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    return [parsed]
            except json.JSONDecodeError:
                continue

        # Try to extract JSON arrays
        array_pattern = r'\[[\s\S]*\]'
        array_matches = re.findall(array_pattern, response_text)

        for match in array_matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                continue

        logger.warning("Could not extract valid JSON from response: %s...", 
                      response_text[:500])
        return []



    async def analyze_policy(self, policy_data: dict) -> dict[str, Any]:
        """
        Complete policy analysis: extract criteria AND generate questionnaire in one streamlined operation.
        
        Args:
            policy_data: Dict containing policy_name, policy_text, and metadata
        
        Returns:
            Dict with both extracted criteria and generated questionnaire
        """
        try:
            # Initialize cost tracking for this specific request (ensures concurrency safety)
            call_costs = []
            
            policy_name = policy_data.get("policy_name", "Unknown Policy")
            policy_text = policy_data.get("policy_text", "")
            page_references = policy_data.get("page_references", [])
            chunk_metadata = policy_data.get("chunk_metadata", {})
            
            # Detect if this is chunk processing vs full document
            is_chunk_processing = "Chunk" in policy_name
            target_questions = "8-12" if is_chunk_processing else "15-25"
            
            # Extract page range information for chunk processing
            page_range_info = ""
            if is_chunk_processing and chunk_metadata:
                start_page = chunk_metadata.get("start_page", 1)
                end_page = chunk_metadata.get("end_page", 1)
                if start_page == end_page:
                    page_range_info = f"This chunk covers page {start_page}."
                else:
                    page_range_info = f"This chunk covers pages {start_page}-{end_page}."
            elif page_references:
                if len(page_references) == 1:
                    page_range_info = f"This document is page {page_references[0]}."
                else:
                    page_range_info = f"This document covers pages {min(page_references)}-{max(page_references)}."
            
            logger.info(f"Analyzing policy '{policy_name}' ({len(policy_text)} characters)")
            logger.info(f"Processing mode: {'Chunk' if is_chunk_processing else 'Full Document'} - Target questions: {target_questions}")
            if page_range_info:
                logger.info(f"Page range: {page_range_info}")
            if chunk_metadata:
                logger.info(f"Chunk metadata: {chunk_metadata}")
            logger.info(f"Page references: {page_references}")
            
            if not policy_text.strip():
                return {
                    "status": "error",
                    "message": "No policy text provided for analysis",
                    "data": {}
                }

            # FILTER OUT NON-MEDICAL CONTENT BEFORE ANALYSIS
            logger.info("Filtering policy text to remove references, bibliography, and administrative sections...")
            original_length = len(policy_text)
            filtered_policy_text = await self._filter_medical_content(policy_text, call_costs)
            filtered_length = len(filtered_policy_text)
            
            # Skip LLM call if filtering removed all content to save API costs
            if filtered_length == 0:
                logger.info("Content marked for complete removal by LLM. Skipping question generation to save API costs.")
                return {
                    "status": "completed",
                    "message": "No medical content found in this section. Skipped question generation.",
                    "data": {
                        "questions": [],
                        "total_questions": 0,
                        "cost_info": {
                            "total_cost": sum(cost.total_cost for cost in call_costs),
                            "input_tokens": sum(cost.input_tokens for cost in call_costs),
                            "output_tokens": sum(cost.output_tokens for cost in call_costs),
                            "model": self.cost_tracker.model,
                            "call_count": len(call_costs)
                        }
                    }
                }
            
            # Use filtered text for analysis (no over-filtering protection)
            reduction_percent = ((original_length - filtered_length) / original_length) * 100
            logger.info(f"Content filtering complete: {filtered_length}/{original_length} characters remaining ({reduction_percent:.1f}% reduction)")
            policy_text = filtered_policy_text

            # Enhanced prompt for structured question generation with answer options
            system_prompt = """You are a senior medical policy analyst specializing in PRIOR AUTHORIZATION requirements for healthcare procedures and treatments.

Your task is to analyze FILTERED medical policy content that has been pre-processed to remove references, bibliography, revision history, and administrative sections. You will ONLY receive the core medical and clinical content needed for creating authorization questions.

IMPORTANT: The policy text you receive has been FILTERED to contain only medical criteria, clinical indications, and eligibility requirements. Do NOT create questions about references, citations, or administrative content.

CRITICAL INSTRUCTIONS:
1. Extract the MOST CRITICAL prior authorization criteria that determine approval/denial
2. For CHUNKED processing: Generate 8-12 focused questions per chunk
3. For SINGLE document processing: Generate 15-25 comprehensive questions  
4. Each question must have predefined answer options that match real-world medical scenarios
5. Include precise source references (page numbers and exact policy text)
6. Focus EXCLUSIVELY on medical criteria that DIRECTLY impact authorization decisions
7. IGNORE any remaining administrative content, revision notes, or citation fragments

PRIOR AUTHORIZATION SPECIFIC FOCUS AREAS:

MEDICAL NECESSITY CRITERIA:
- Specific medical conditions/diagnoses (with ICD-10 codes where mentioned)
- Disease severity indicators and thresholds
- Failed conservative treatments or step therapy requirements
- Symptom duration and progression requirements
- Comorbidity requirements and contraindications

CLINICAL ELIGIBILITY:
- Age requirements (minimum/maximum ages)
- BMI thresholds for procedures like bariatric surgery
- Lab values and diagnostic test requirements
- Physical/psychological evaluation requirements
- Pregnancy status and reproductive considerations

DOCUMENTATION REQUIREMENTS:
- Required physician specialties (surgeon, cardiologist, etc.)
- Specific medical records and timeframes
- Required diagnostic imaging or tests
- Pre-authorization forms and attestations
- Insurance coverage verification requirements

PROCEDURAL SPECIFICATIONS:
- Covered vs. non-covered procedure codes
- Facility requirements (inpatient vs. outpatient)
- Surgeon qualification requirements
- Equipment or technique specifications

EXCLUSION CRITERIA:
- Absolute medical contraindications
- Age-related exclusions
- Specific diagnoses that disqualify
- Previous procedure history limitations

QUESTION TYPES AND ANSWER OPTIONS:
- yes_no: ["Yes", "No", "Not Documented"]
- yes_no_unknown: ["Yes", "No", "Unknown", "Not Documented"]
- numeric: {"type": "numeric", "min": X, "max": Y, "unit": "kg/years/months", "required": true}
- text: {"type": "text", "format": "ICD-10 code|procedure code|diagnosis|free text", "max_length": 500}
- date: {"type": "date", "format": "MM/DD/YYYY", "required": true}
- multiple_choice: Provide 3-6 specific medical options relevant to the criterion
- duration: {"type": "duration", "unit": "months|weeks|years", "min": X, "max": Y}

QUALITY REQUIREMENTS:
- Questions must be medically accurate and clinically relevant
- Avoid redundant questions - each should test a unique criterion
- Focus on deterministic criteria that lead to clear approve/deny decisions
- Include both qualifying criteria AND disqualifying exclusions
- Questions should be answerable from typical medical records

EXAMPLES OF HIGH-QUALITY QUESTIONS:

GOOD EXAMPLES:
✓ "Does the patient have a diagnosis of morbid obesity (ICD-10 code E66.01 or E66.09)?" (specific diagnosis with codes)
✓ "What is the patient's current Body Mass Index (BMI)?" (measurable criterion with numeric input)
✓ "Has the patient failed conservative treatment for at least 6 months?" (specific duration requirement)
✓ "Is the patient aged 18 years or older?" (clear age requirement)
✓ "Does the patient have any of the following obesity-related comorbid conditions: diabetes, hypertension, sleep apnea, cardiovascular disease?" (specific comorbidities)

AVOID THESE EXAMPLES:
✗ "Are there any references cited in this policy?" (administrative content)
✗ "What year was this policy last revised?" (document history)
✗ "Who should be contacted for questions about this policy?" (contact information)
✗ "What website provides additional information?" (external resources)
✗ "Is this policy published in a medical journal?" (publication details)

Return ONLY a JSON object with this exact format:
{{
  "questionnaire_metadata": {{
    "policy_name": "policy name",
    "total_questions": 0,
    "complexity_level": "simple|moderate|complex",
    "coverage_areas": ["medical_necessity", "clinical_eligibility", "documentation", "procedural_requirements", "exclusions"],
    "procedure_type": "bariatric_surgery|cardiac|oncology|orthopedic|other",
    "target_population": "description of patient population"
  }},
  "questions": [
    {{
      "question_id": "unique_id_like_BMI_001_or_AGE_001",
      "question_text": "Clear, specific medical question",
      "question_type": "yes_no|yes_no_unknown|numeric|text|date|multiple_choice|duration",
      "answer_options": ["option1", "option2"] or {{"type": "numeric", "min": 18, "max": 65, "unit": "years", "required": true}},
      "source_text_snippet": "Exact policy text that supports this criterion (50-300 chars)",
      "page_number": "Specific page number where this criterion is found",
      "line_reference": "section or paragraph reference",
      "criterion_type": "medical_necessity|clinical_eligibility|documentation|procedural_requirements|exclusions",
      "priority_level": "high|medium|low",
      "validation_rules": ["rule1", "rule2"],
      "approval_impact": "required_for_approval|preferred|informational"
    }}
  ]
}}"""

            user_prompt = f"""FILTERED MEDICAL POLICY CONTENT: {policy_name}
PROCESSING MODE: {'CHUNK PROCESSING' if is_chunk_processing else 'FULL DOCUMENT'}
TARGET QUESTIONS: {target_questions}
{page_range_info}

IMPORTANT: The content below has been FILTERED to remove references, bibliography, revision history, and administrative sections. You are receiving ONLY the medical and clinical content relevant for creating authorization questions.

CONTENT FILTERING APPLIED:
- Removed references, citations, and bibliography sections
- Removed revision history and document updates
- Removed administrative content and contact information
- Removed website links and external resources
- Retained ONLY medical criteria, clinical indications, and eligibility requirements

When setting page_number in questions, analyze the text content to find the specific page where the criterion was found.
Look for page markers like "=== PAGE X ===" in the text and set the page_number to the specific page that contains the relevant content.
If a criterion spans multiple pages, use the primary page where the key information is located.

FILTERED MEDICAL CONTENT:
{policy_text}

Analyze this FILTERED medical content and create a {'focused' if is_chunk_processing else 'comprehensive'} prior authorization questionnaire. 

FOCUS EXCLUSIVELY ON MEDICAL CRITERIA - IGNORE ANY REMAINING ADMINISTRATIVE CONTENT:

MEDICAL NECESSITY CRITERIA (HIGH PRIORITY):
- Primary medical conditions/diagnoses requiring the procedure (e.g., morbid obesity for bariatric surgery)
- Specific ICD-10 diagnosis codes mentioned in policy (E66.01, E66.09, etc.)
- Disease severity indicators (BMI >40, failed medical management for >6 months)
- Comorbidity requirements (diabetes, hypertension, sleep apnea, cardiovascular disease)
- Failed conservative treatment documentation requirements (duration, types of treatment)
- Symptom duration and progression requirements (timeline of medical management)

CLINICAL ELIGIBILITY REQUIREMENTS (HIGH PRIORITY):
- Age requirements (e.g., 18-65 years for bariatric surgery)
- BMI thresholds and weight criteria (e.g., BMI ≥40 or BMI ≥35 with comorbidities)
- Physical assessment requirements
- Psychological evaluation requirements
- Pregnancy status and reproductive planning
- Life expectancy considerations

DOCUMENTATION AND EVALUATION REQUIREMENTS (MEDIUM PRIORITY):
- Required specialist evaluations (surgeon, nutritionist, psychologist)
- Specific medical records and timeframes (6-12 months of documented treatment)
- Required diagnostic tests and imaging
- Pre-authorization forms and physician attestations
- Insurance coverage verification requirements
- Patient education and informed consent documentation

PROCEDURAL AND FACILITY REQUIREMENTS (MEDIUM PRIORITY):
- Specific covered procedure codes (CPT codes)
- Facility accreditation requirements (Center of Excellence designation)
- Surgeon qualification and experience requirements
- Equipment and technique specifications
- Inpatient vs. outpatient setting requirements

EXCLUSIONS AND CONTRAINDICATIONS (HIGH PRIORITY):
- Absolute medical contraindications (active substance abuse, untreated psychiatric conditions)
- Age-related exclusions (too young or elderly)
- Previous procedure history that disqualifies
- Medical conditions that increase surgical risk
- Non-compliance with pre-operative requirements

QUESTION PRIORITIZATION GUIDANCE:
- If processing a CHUNK: Generate 8-12 most critical questions (focus on quality over quantity)
- If processing FULL document: Generate 15-25 comprehensive questions
- PRIORITY ORDER: Medical necessity → Clinical eligibility → Exclusions → Documentation → Procedural

QUESTION TYPE SELECTION WITH COMPLETE ANSWER OPTIONS:

1. yes_no: Binary medical decisions
   - Answer options: ["Yes", "No", "Not Documented"]
   - Use for: Has diagnosis, completed treatment, meets criteria

2. numeric: Measurements and thresholds
   - Answer options: {{"type": "numeric", "min": 0, "max": 100, "unit": "kg/m²", "required": true}}
   - Use for: BMI, age, weight, lab values
   - ALWAYS include min, max, unit, and required fields

3. multiple_choice: Categorical selections
   - Answer options: ["Option 1", "Option 2", "Option 3", "None of the above"]
   - Use for: Comorbidities, surgery types, procedure codes
   - Include 3-6 relevant medical options plus "None of the above"

4. text: Open-ended medical information
   - Answer options: {{"type": "text", "format": "ICD-10 code", "max_length": 50, "required": true}}
   - Use for: Specific codes, diagnoses, medications
   - Specify format and max_length

5. date: Timeline requirements
   - Answer options: {{"type": "date", "format": "MM/DD/YYYY", "required": true}}
   - Use for: Diagnosis date, treatment start/end dates

6. duration: Time-based requirements
   - Answer options: {{"type": "duration", "unit": "months", "min": 1, "max": 24, "required": true}}
   - Use for: Treatment duration, symptom timeline

MEDICAL ACCURACY REQUIREMENTS:
- Questions must be clinically relevant and medically sound
- Use proper medical terminology consistently
- Include specific thresholds mentioned in policy (exact BMI, age ranges)
- Reference specific medical specialties when required
- Include both inclusion criteria (what qualifies) AND exclusion criteria (what disqualifies)

Each question should directly contribute to an approve/deny decision. Avoid informational questions that don't impact authorization.
Return ONLY the JSON object."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            logger.info("Sending policy analysis request to LLM")
            response = await self.model.ainvoke(messages)
            
            # Track cost for this LLM call (using request-local list for concurrency safety)
            cost_data = self.cost_tracker.calculate_cost(response)
            call_costs.append(cost_data)
            
            # Parse the enhanced structured response
            result_data = self._extract_json_from_response(response.content)
            
            if not result_data or not isinstance(result_data[0], dict):
                logger.error("Invalid response format from LLM")
                return {
                    "status": "error",
                    "message": "Failed to parse LLM response",
                    "data": {}
                }
            
            analysis_result = result_data[0]
            questionnaire_metadata = analysis_result.get("questionnaire_metadata", {})
            questions = analysis_result.get("questions", [])
            
            # Validate question count based on processing mode
            if is_chunk_processing:
                min_questions = 5
                max_questions = 12
            else:
                min_questions = 10
                max_questions = 25
            
            if len(questions) < min_questions:
                logger.warning(f"Generated only {len(questions)} questions, minimum is {min_questions} for {'chunk' if is_chunk_processing else 'full document'}")
            elif len(questions) > max_questions:
                logger.warning(f"Generated {len(questions)} questions, maximum is {max_questions} for {'chunk' if is_chunk_processing else 'full document'}, truncating")
                questions = questions[:max_questions]
            
            logger.info(f"Analysis complete: {len(questions)} questions generated")

            # Aggregate cost information (using request-local list for concurrency safety)
            aggregated_costs = self.cost_tracker.aggregate_costs(call_costs)
            
            result = {
                "status": "completed",
                "message": f"Successfully analyzed policy '{policy_name}': {len(questions)} questions generated",
                "data": {
                    "policy_name": policy_name,
                    "questionnaire_metadata": questionnaire_metadata,
                    "questions": questions,
                    "page_references": page_references,
                    "summary": {
                        "questions_count": len(questions),
                        "policy_length": len(policy_text),
                        "complexity_level": questionnaire_metadata.get("complexity_level", "moderate"),
                        "coverage_areas": questionnaire_metadata.get("coverage_areas", [])
                    },
                    "cost_info": aggregated_costs
                }
            }
            
            # MEMORY CLEANUP: Clear large variables that may accumulate across chunk calls
            try:
                del policy_text, filtered_policy_text, system_prompt, user_prompt
                if 'messages' in locals():
                    del messages
                if 'response' in locals():
                    del response
                gc.collect()
            except:
                pass  # Ignore cleanup errors
                
            return result

        except Exception as e:
            logger.error(f"Error during policy analysis: {e}", exc_info=True)
            
            # MEMORY CLEANUP: Clear variables even on error
            try:
                if 'policy_text' in locals():
                    del policy_text
                if 'filtered_policy_text' in locals():
                    del filtered_policy_text
                if 'system_prompt' in locals():
                    del system_prompt
                if 'user_prompt' in locals():
                    del user_prompt
                if 'messages' in locals():
                    del messages
                if 'response' in locals():
                    del response
                gc.collect()
            except:
                pass  # Ignore cleanup errors
                
            return {
                "status": "error",
                "message": f"Policy analysis error: {str(e)}",
                "data": {}
            }

    async def consolidate_questionnaire(self, consolidation_data: dict, call_costs: list = None) -> dict[str, Any]:
        """
        Consolidate multiple questionnaire chunks into a final optimized questionnaire.
        Removes duplicates, merges similar questions, and ensures optimal question count.
        
        Args:
            consolidation_data: Dict containing list of question chunks to consolidate
            call_costs: List to track costs for this request (for concurrency safety)
            
        Returns:
            Dict with consolidated questionnaire
        """
        try:
            # Initialize cost tracking if not provided (for backward compatibility)
            if call_costs is None:
                call_costs = []
                
            question_chunks = consolidation_data.get("question_chunks", [])
            policy_name = consolidation_data.get("policy_name", "Unknown Policy")
            target_question_count = consolidation_data.get("target_question_count", 20)
            
            logger.info(f"Consolidating {len(question_chunks)} question chunks for '{policy_name}'")
            
            if not question_chunks:
                return {
                    "status": "error",
                    "message": "No question chunks provided for consolidation",
                    "data": {}
                }
            
            # Create summary of questions for more efficient processing
            question_summaries = []
            for chunk_idx, chunk in enumerate(question_chunks):
                chunk_questions = chunk.get("questions", [])
                chunk_metadata = chunk.get("chunk_metadata", {})
                chunk_start_page = chunk_metadata.get("start_page", 1)
                chunk_end_page = chunk_metadata.get("end_page", 1)
                
                for q_idx, question in enumerate(chunk_questions):
                    # Use chunk page range instead of individual question page number
                    page_reference = question.get("page_number", chunk_start_page)
                    if chunk_start_page != chunk_end_page:
                        page_reference = f"{chunk_start_page}-{chunk_end_page}"
                    
                    summary = {
                        "id": f"chunk_{chunk_idx}_q_{q_idx}",
                        "text": question.get("question_text", "")[:200],  # Truncate long text
                        "type": question.get("question_type", "text"),
                        "category": question.get("criterion_type", "general"),
                        "page": page_reference,
                        "chunk_pages": f"{chunk_start_page}-{chunk_end_page}" if chunk_start_page != chunk_end_page else str(chunk_start_page),
                        "answer_options": question.get("answer_options", [])
                    }
                    logger.debug(f"Question {chunk_idx}_{q_idx}: original_page={question.get('page_number')}, chunk_pages={chunk_start_page}-{chunk_end_page}, final_page={page_reference}")
                    question_summaries.append(summary)
            
            logger.info(f"Total questions before consolidation: {len(question_summaries)}")
            
            system_prompt = """You are a medical questionnaire optimization specialist with expertise in prior authorization requirements.

Your task is to consolidate question summaries into the best possible questionnaire for prior authorization decisions.

CRITICAL DEDUPLICATION RULES:
1. IDENTIFY DUPLICATES: Questions asking the same thing in different words
   - "Has the patient failed conservative treatment for 6 months?" 
   - "Has the patient provided documentation of 6 months of medical management?"
   - These are essentially the same - merge into one comprehensive question

2. MERGE SIMILAR QUESTIONS: Combine related questions into comprehensive versions
   - Multiple BMI-related questions → One comprehensive BMI question
   - Multiple age-related questions → One age eligibility question
   - Multiple comorbidity questions → One comprehensive comorbidity checklist

3. PRIORITIZE HIGH-VALUE QUESTIONS:
   - Questions that directly determine approval/denial (HIGH PRIORITY)
   - Questions required by policy criteria (HIGH PRIORITY)  
   - Documentation questions (MEDIUM PRIORITY)
   - Informational questions (LOW PRIORITY - often remove)

4. MAINTAIN COVERAGE: Ensure all critical areas are covered:
   - Medical necessity criteria
   - Clinical eligibility requirements
   - Exclusion criteria
   - Documentation requirements
   - Procedural requirements

5. OPTIMIZE QUESTION COUNT: Target {target_question_count} high-quality questions

INPUT FORMAT: You'll receive question summaries with id, text, type, category, priority, page, chunk_pages.
OUTPUT: Return complete consolidated questions ready for use.
IMPORTANT: Use 'page_references' instead of 'page_number' - set it to a string like "1-3" or array like [1,2,3] for questions spanning multiple pages.

Return ONLY a JSON object with this exact format:
{{
  "consolidated_questionnaire": {{
    "questions": [
      {{
        "question_id": "q1",
        "question_text": "Complete question text",
        "question_type": "yes_no",
        "answer_options": ["Yes", "No", "N/A"],
        "criterion_type": "eligibility",
        "page_references": [1, 2, 3] or "1-3",
        "source_ids": ["original_id_1", "original_id_2"]
      }}
    ],
    "total_questions": 0,
    "optimization_summary": "brief description of consolidation approach"
  }}
}}"""
            
            user_prompt = f"""POLICY: {policy_name}
TARGET QUESTION COUNT: {target_question_count}

QUESTION SUMMARIES TO CONSOLIDATE:
{json.dumps(question_summaries, indent=2)}

Consolidate these questions into an optimized questionnaire of approximately {target_question_count} questions.

INSTRUCTIONS:
1. Review all question summaries
2. Remove duplicates and similar questions
3. Merge related questions into comprehensive versions
4. Select the most critical questions and ensure diverse coverage
5. Create complete question objects with all required fields
6. Ensure final count is around {target_question_count}

IMPORTANT: Return complete question objects in the "questions" array, not just IDs or references.

Return ONLY the JSON object in the exact format specified."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            logger.info("Sending consolidation request to LLM")
            response = await self.model.ainvoke(messages)
            
            # Track cost for this LLM call (using request-local list for concurrency safety)
            cost_data = self.cost_tracker.calculate_cost(response)
            call_costs.append(cost_data)
            
            result_data = self._extract_json_from_response(response.content)
            
            if not result_data or not isinstance(result_data[0], dict):
                logger.error("Invalid consolidation response format from LLM")
                return {
                    "status": "error",
                    "message": "Failed to parse consolidation response",
                    "data": {}
                }
            
            consolidation_result = result_data[0]
            consolidated_questionnaire = consolidation_result.get("consolidated_questionnaire", {})
            
            # Extract final questions from the consolidation result
            final_questions = consolidated_questionnaire.get("questions", [])
            final_question_count = len(final_questions)
            
            if not final_questions:
                logger.error("No questions found in consolidation result")
                logger.error(f"Available keys: {list(consolidated_questionnaire.keys())}")
                return {
                    "status": "error",
                    "message": "Consolidation failed - no questions returned",
                    "data": {
                        "available_keys": list(consolidated_questionnaire.keys()),
                        "raw_response": consolidated_questionnaire
                    }
                }
            logger.info(f"Consolidation complete: {len(question_summaries)} -> {final_question_count} questions")
            
            # Return the final questions in the expected format
            result = {
                "status": "completed",
                "message": f"Successfully consolidated questionnaire for '{policy_name}': {final_question_count} final questions",
                "data": {
                    "policy_name": policy_name,
                    "consolidated_questionnaire": {
                        "questions": final_questions,  # Make sure questions are in the expected key
                        "total_questions": final_question_count,
                        "optimization_summary": consolidated_questionnaire.get("optimization_summary", "Questions consolidated successfully"),
                        "original_response": consolidated_questionnaire  # Keep original for debugging
                    },
                    "consolidation_stats": {
                        "original_count": len(question_summaries),
                        "final_count": final_question_count,
                        "reduction_percentage": round((1 - final_question_count/len(question_summaries)) * 100, 1) if question_summaries else 0
                    }
                }
            }
            
            # MEMORY CLEANUP: Clear large consolidation variables
            try:
                del question_summaries, system_prompt, user_prompt, messages, response
                if 'result_data' in locals():
                    del result_data
                if 'consolidation_result' in locals():
                    del consolidation_result
                gc.collect()
            except:
                pass
                
            return result
            
        except Exception as e:
            logger.error(f"Error during questionnaire consolidation: {e}", exc_info=True)
            
            # MEMORY CLEANUP: Clear variables even on error
            try:
                if 'question_summaries' in locals():
                    del question_summaries
                if 'system_prompt' in locals():
                    del system_prompt
                if 'user_prompt' in locals():
                    del user_prompt
                if 'messages' in locals():
                    del messages
                if 'response' in locals():
                    del response
                gc.collect()
            except:
                pass
            
            return {
                "status": "error",
                "message": f"Consolidation error: {str(e)}",
                "data": {}
            }

    async def invoke(self, task: str, input_data: dict) -> dict[str, Any]:
        """
        Enhanced invoke method supporting multiple tasks.

        Args:
            task: Task type ('analyze_policy' or 'consolidate_questionnaire')
            input_data: Input data for the task

        Returns:
            Dict containing the response
        """
        logger.info(f"Invoking agent for task: {task}")

        try:
            if task == "analyze_policy":
                return await self.analyze_policy(input_data)
            elif task == "consolidate_questionnaire":
                return await self.consolidate_questionnaire(input_data)
            else:
                return {
                    "status": "error",
                    "message": f"Unknown task type: {task}. Supported: analyze_policy, consolidate_questionnaire",
                    "data": {}
                }

        except Exception as e:
            logger.error(f"Error during invoke: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Agent error: {str(e)}",
                "data": {}
            }