"""
Policy extraction engine with hierarchy detection.
Supports both legacy DocumentChunk objects and chunk dictionaries.
"""
import json
import re
from typing import List, Dict, Any, Optional, Union
from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from openai import InternalServerError, APITimeoutError
from settings import settings
from utils.logger import get_logger
from core.data_models import DocumentChunk, PDFPage
from models.schemas import (
    SubPolicy,
    PolicyCondition,
    PolicyHierarchy,
    SourceReference,
)

logger = get_logger(__name__)


def _get_chunk_attr(chunk: Union[Dict, Any], attr: str, default: Any = None) -> Any:
    """
    Get attribute from chunk, supporting both dict and object formats.
    
    Args:
        chunk: Chunk as dict or object
        attr: Attribute name
        default: Default value if not found
    
    Returns:
        Attribute value
    """
    if isinstance(chunk, dict):
        return chunk.get(attr, default)
    return getattr(chunk, attr, default)


class PolicyExtractor:
    """Extracts policies and sub-policies from documents with hierarchy detection."""

    def __init__(self, use_gpt4: bool = False, llm: Optional[ChatOpenAI] = None):
        """
        Initialize policy extractor.

        Args:
            use_gpt4: Whether to use GPT-4 for extraction
            llm: Optional pre-configured LLM client
        """
        self.llm = llm
        self.use_gpt4 = use_gpt4
        model_name = "gpt-4o" if use_gpt4 else "gpt-4o-mini"
        logger.info(f"Policy extractor initialized with model: {model_name}")
        logger.debug(f"Max concurrent requests: {settings.openai_max_concurrent_requests}")

    async def extract_policies(
        self, 
        chunks: List[Union[Dict, DocumentChunk]], 
        pages: List[Union[Dict, PDFPage]],
        doc_metadata: Optional[Dict[str, Any]] = None
    ) -> PolicyHierarchy:
        """
        Extract all policies and their hierarchy from document chunks in parallel.

        Args:
            chunks: List of chunk dicts or DocumentChunk objects
            pages: List of page dicts or PDFPage objects
            doc_metadata: Optional document metadata for filtering/context

        Returns:
            PolicyHierarchy object
        """
        import asyncio

        # Filter chunks if metadata available
        chunks_to_process = self._filter_chunks_for_extraction(chunks, doc_metadata)
        
        logger.info(f"Policy extraction: {len(chunks_to_process)}/{len(chunks)} chunks to process")
        
        # Create semaphore to limit concurrent API requests
        semaphore = asyncio.Semaphore(settings.openai_max_concurrent_requests)

        model_name = "gpt-4o" if self.use_gpt4 else "gpt-4o-mini"
        logger.info(f"Using model: {model_name} with max {settings.openai_max_concurrent_requests} concurrent requests")

        async def extract_with_semaphore(chunk, pages_list, index, total):
            async with semaphore:
                return await self._extract_chunk_with_timeout(chunk, pages_list, index, total)

        # Create tasks for all chunks
        tasks = []
        for i, chunk in enumerate(chunks_to_process):
            tasks.append(extract_with_semaphore(chunk, pages, i + 1, len(chunks_to_process)))

        # Run all tasks with concurrency control
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all policies and definitions
        all_policies = []
        definitions = {}
        errors_count = 0
        timeout_errors = 0
        failed_chunks = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors_count += 1
                error_type = type(result).__name__
                
                # Track specific error types
                if "Timeout" in error_type or "timeout" in str(result).lower():
                    timeout_errors += 1
                
                # Store chunk info for failed extractions
                chunk_id = _get_chunk_attr(chunks_to_process[i], "chunk_id", f"chunk_{i}")
                failed_chunks.append(chunk_id)
                
                logger.error(f"Error extracting from chunk {i + 1} ({chunk_id}): [{error_type}] {result}")
                continue

            if result is None:
                logger.warning(f"Chunk {i + 1} returned no results")
                continue

            chunk_policies, chunk_definitions = result
            all_policies.extend(chunk_policies)
            definitions.update(chunk_definitions)

        logger.info(
            f"Extraction complete: {len(all_policies)} policies, "
            f"{len(definitions)} definitions from {len(chunks_to_process) - errors_count}/{len(chunks_to_process)} chunks"
        )
        if errors_count > 0:
            logger.warning(f"Failed chunks: {errors_count} ({', '.join(failed_chunks)})")
        if timeout_errors > 0:
            logger.warning(f"Timeout failures: {timeout_errors} - Consider increasing openai_per_request_timeout or reducing concurrent requests")

        # Build hierarchy
        logger.info("Building policy hierarchy from extracted policies...")
        policy_hierarchy = self._build_hierarchy(all_policies, definitions)

        logger.info(
            f"Policy extraction complete: {policy_hierarchy.total_policies} policies, "
            f"{len(policy_hierarchy.root_policies)} root, depth={policy_hierarchy.max_depth}"
        )

        return policy_hierarchy

    def _filter_chunks_for_extraction(
        self, 
        chunks: List[Union[Dict, Any]], 
        doc_metadata: Optional[Dict[str, Any]]
    ) -> List[Union[Dict, Any]]:
        """
        Filter chunks for extraction using metadata.
        Skip chunks with low context completeness or non-policy content.
        
        Args:
            chunks: List of chunks
            doc_metadata: Document metadata
        
        Returns:
            Filtered list of chunks
        """
        if not doc_metadata:
            return chunks
        
        # Filter based on chunk metadata
        filtered = []
        for chunk in chunks:
            metadata = _get_chunk_attr(chunk, "metadata", {})
            
            # CRITICAL FIX: Skip chunks marked as duplicates
            if metadata.get("is_duplicate", False):
                chunk_id = _get_chunk_attr(chunk, "chunk_id", "unknown")
                duplicate_of = metadata.get("duplicate_of", "unknown")
                similarity = metadata.get("duplicate_similarity", 0)
                logger.info(
                    f"Skipping chunk {chunk_id}: marked as duplicate of {duplicate_of} "
                    f"(similarity={similarity:.2f})"
                )
                continue
            
            # Skip if explicitly marked as non-policy
            has_complete_context = metadata.get("has_complete_context", True)
            boundary_type = metadata.get("boundary_type", "")
            
            # Include chunks that have complete context or are policy boundaries
            if has_complete_context or boundary_type in ["policy", "content_zone"]:
                filtered.append(chunk)
            else:
                chunk_id = _get_chunk_attr(chunk, "chunk_id", "unknown")
                logger.debug(f"Skipping chunk {chunk_id}: incomplete context")
        
        if len(filtered) < len(chunks):
            logger.info(f"Filtered {len(chunks) - len(filtered)} chunks without complete context")
        
        return filtered

    async def _extract_chunk_with_timeout(
        self, chunk: Union[Dict, Any], pages: List[Union[Dict, Any]], index: int, total: int
    ) -> tuple[List[SubPolicy], Dict[str, str]]:
        """
        Extract from chunk with timeout.

        Args:
            chunk: Chunk dict or object
            pages: List of page dicts or objects
            index: Current chunk index
            total: Total number of chunks

        Returns:
            Tuple of (policies, definitions) or None on timeout
        """
        import asyncio

        try:
            start_page = _get_chunk_attr(chunk, "start_page", 0)
            end_page = _get_chunk_attr(chunk, "end_page", 0)
            text = _get_chunk_attr(chunk, "text", "")
            
            logger.debug(f"[Chunk {index}/{total}] Extracting (Pages {start_page}-{end_page}, {len(text)} chars)")

            result = await asyncio.wait_for(
                self._extract_from_chunk(chunk, pages),
                timeout=settings.openai_per_request_timeout
            )

            policies, defs = result
            logger.info(f"[Chunk {index}/{total}] Complete: {len(policies)} policies, {len(defs)} definitions")
            return result

        except asyncio.TimeoutError:
            logger.error(f"[Chunk {index}/{total}] Timeout after {settings.openai_per_request_timeout}s")
            return [], {}

        except Exception as e:
            logger.error(f"[Chunk {index}/{total}] Error: {e}", exc_info=True)
            return [], {}

    @retry(
        stop=stop_after_attempt(settings.openai_max_retries),
        wait=wait_exponential(
            multiplier=settings.openai_retry_multiplier,
            min=settings.openai_retry_min_wait,
            max=settings.openai_retry_max_wait
        ),
        retry=retry_if_exception_type((InternalServerError, APITimeoutError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logger.level),
        reraise=True,
    )
    async def _extract_from_chunk(
        self, chunk: Union[Dict, Any], pages: List[Union[Dict, Any]]
    ) -> tuple[List[SubPolicy], Dict[str, str]]:
        """
        Extract policies from a single chunk.

        Args:
            chunk: Chunk dict or object
            pages: List of page dicts or objects

        Returns:
            Tuple of (list of SubPolicy objects, definitions dict)
        """
        prompt = self._create_extraction_prompt(chunk)
        chunk_id = _get_chunk_attr(chunk, "chunk_id", "unknown")

        try:
            if self.llm is None:
                logger.error(f"LLM not initialized for chunk {chunk_id}")
                return [], {}

            logger.debug(f"Calling LLM for chunk {chunk_id}")

            response = await self.llm.ainvoke(prompt)

            logger.debug(f"Received response for chunk {chunk_id}")

            # Parse response
            extraction_result = self._parse_extraction_result(response.content, chunk)

            policies = extraction_result.get("policies", [])
            definitions = extraction_result.get("definitions", {})

            logger.debug(f"Parsed {len(policies)} policies, {len(definitions)} definitions from chunk {chunk_id}")

            return policies, definitions

        except (InternalServerError, APITimeoutError) as e:
            error_type = type(e).__name__
            logger.error(f"LiteLLM proxy error for chunk {chunk_id} ({error_type}): {e}")
            raise  # Let tenacity retry handle it
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for chunk {chunk_id}: {e}")
            return [], {}
        except Exception as e:
            logger.error(f"Error extracting from chunk {chunk_id}: {e}")
            raise  # Let tenacity retry handle it

    def _create_extraction_prompt(self, chunk: Union[Dict, Any]) -> str:
        """
        Create extraction prompt for a chunk.

        Args:
            chunk: Chunk dict or object

        Returns:
            Prompt string
        """
        start_page = _get_chunk_attr(chunk, "start_page", 0)
        end_page = _get_chunk_attr(chunk, "end_page", 0)
        text = _get_chunk_attr(chunk, "text", "")
        
        prompt = f"""You are an expert at analyzing insurance policy documents. Before extracting policies, understand these key principles:

=== INSURANCE POLICY DOCUMENT STRUCTURE ===
1. **Document Hierarchy**:
   - Major Policy Sections (Level 0): Top-level coverage areas
     * Examples: "Medical Necessity Criteria", "Coverage Guidelines", "Exclusions and Limitations"
     * Typically numbered with Roman numerals (I, II, III) or large numbers (1., 2., 3.)
     * Contains SUBSTANTIAL content (multiple paragraphs, several pages)
   
   - Sub-Policies (Level 1): Specific criteria within major sections
     * Examples: "Adolescent Eligibility", "BMI Requirements", "Pre-Authorization Process"
     * Typically lettered (A., B., C.) or decimal numbered (1.1, 1.2)
     * Contains DISTINCT eligibility/coverage criteria from parent
   
   - Sub-Sub-Policies (Level 2): Rare, only for deeply nested requirements
     * Examples: "Age-specific BMI thresholds for adolescents"
     * Only create if clearly nested under level 1 policy

2. **What IS a Policy**:
   ✅ DO EXTRACT:
   - Complete coverage criteria with eligibility requirements
   - Major sections with multiple conditions/requirements
   - Distinct sub-requirements that independently determine eligibility
   - Self-contained rules with specific thresholds/values
   
   ❌ DO NOT EXTRACT:
   - Individual bullet points (group under parent policy)
   - Examples or clarifications (include in parent description)
   - Administrative text (cover pages, disclaimers, TOC)
   - Redundant restatements (extract once in most relevant place)
   - Single sentences without actionable criteria

3. **Hierarchy Detection Rules**:
   - If heading has Roman numeral (I, II) or number+ALL CAPS → Level 0
   - If heading has letter (A., B.) or decimal (1.1, 1.2) → Level 1
   - If content is under another policy section → Set parent_id correctly
   - If unsure about parent, use context (what section are we in?)

4. **Naming Convention**:
   - Use SEMANTIC, DESCRIPTIVE IDs: 'bariatric_surgery_bmi_criteria' ✅
   - NOT generic IDs: 'policy_1', 'criteria_section' ❌
   - For sub-policies: 'parent_name_specific_descriptor' ✅
   - Make each ID UNIQUE and meaningful

=== BEFORE/AFTER EXAMPLES ===

❌ WRONG EXTRACTION (Over-fragmentation):
{{
  "policies": [
    {{"policy_id": "policy_1", "title": "Medical Necessity", "level": 0, "parent_id": null}},
    {{"policy_id": "policy_2", "title": "BMI must be 40+", "level": 0, "parent_id": null}},
    {{"policy_id": "policy_3", "title": "Age 18-65", "level": 0, "parent_id": null}},
    {{"policy_id": "policy_4", "title": "Prior treatments", "level": 0, "parent_id": null}},
    {{"policy_id": "policy_5", "title": "Comorbidities", "level": 0, "parent_id": null}}
  ]
}}
Problems: 11 root policies, no hierarchy, generic IDs, bullet points as policies

✅ CORRECT EXTRACTION (Proper Structure):
{{
  "policies": [
    {{
      "policy_id": "bariatric_surgery_medical_necessity",
      "title": "Medical Necessity Criteria for Bariatric Surgery",
      "description": "Comprehensive eligibility requirements for surgical weight loss procedures",
      "level": 0,
      "parent_id": null,
      "conditions": [
        {{"condition_id": "bmi_threshold", "description": "BMI ≥40 OR BMI 35-39.9 with comorbidities", "logic_type": "OR"}},
        {{"condition_id": "age_requirement", "description": "Patient age between 18-65 years", "logic_type": "AND"}},
        {{"condition_id": "prior_treatment", "description": "Failed 6+ months of supervised weight loss", "logic_type": "AND"}},
        {{"condition_id": "psychological_eval", "description": "Cleared by mental health professional", "logic_type": "AND"}}
      ]
    }},
    {{
      "policy_id": "bariatric_surgery_adolescent_criteria",
      "title": "Adolescent Eligibility Criteria",
      "description": "Modified criteria for patients aged 13-17",
      "level": 1,
      "parent_id": "bariatric_surgery_medical_necessity",
      "conditions": [
        {{"condition_id": "adolescent_bmi", "description": "BMI ≥40 OR BMI 35+ with severe comorbidity", "logic_type": "OR"}},
        {{"condition_id": "adolescent_age", "description": "Age 13-17 with skeletal maturity", "logic_type": "AND"}},
        {{"condition_id": "family_support", "description": "Committed family support system", "logic_type": "AND"}}
      ]
    }}
  ]
}}
Result: 1 root policy with 1 sub-policy, proper hierarchy, semantic IDs, conditions grouped

=== YOUR TASK ===
Extract ALL policies from this document section, following the principles above.

Document Section (Pages {start_page}-{end_page}):
{text}

Extraction Instructions:
1. **Identify Major Sections First** (Level 0):
   - Look for Roman numerals (I, II, III), large numbers (1., 2.), or ALL CAPS headings
   - These should span multiple paragraphs/pages with substantial content
   - Extract as root policies (level=0, parent_id=null)

2. **Identify Sub-Sections** (Level 1):
   - Look for letters (A., B., C.), decimals (1.1, 1.2), or title case headings
   - These should have DISTINCT criteria from their parent section
   - Extract as sub-policies (level=1, parent_id=<major_section_id>)

3. **Create Descriptive IDs**:
   - Use SEMANTIC terms from document: 'bariatric_surgery_coverage', 'adolescent_eligibility'
   - NOT generic: 'policy_1', 'section_a', 'criteria_2'
   - Make each ID uniquely describe its purpose

4. **Group Related Conditions**:
   - Bullet points under same heading → Multiple conditions in ONE policy
   - Similar criteria in different places → Extract ONCE in most relevant policy
   - Don't create separate policies for each bullet point

5. **Set Parent Relationships**:
   - If this section is clearly under another section → Set parent_id
   - Use context clues: indentation, numbering scheme, page structure
   - When in doubt: Check if removal of "parent" makes "child" meaningless

6. **Assign Confidence Scores**:
   - Clear, explicit statements: 0.9-1.0
   - Partially complete info: 0.7-0.9
   - Ambiguous statements: 0.5-0.7

Return JSON in this structure:
{{
  "policies": [
    {{
      "policy_id": "descriptive_snake_case_id",
      "title": "Exact Heading from Document",
      "description": "1-2 sentence summary of what this policy covers",
      "level": 0 or 1 or 2,
      "parent_id": null or "parent_policy_id",
      "conditions": [
        {{
          "condition_id": "descriptive_condition_id",
          "description": "Specific, detailed condition with exact values/thresholds",
          "logic_type": "AND" or "OR" or "NOT" or "SIMPLE",
          "page_number": {start_page},
          "quoted_text": "Exact text from document",
          "confidence_score": 0.95
        }}
      ],
      "page_number": {start_page},
      "section": "Document Section Name",
      "quoted_text": "Exact representative text from source",
      "confidence_score": 0.9
    }}
  ],
  "definitions": {{
    "Term": "Definition (only if explicitly defined in document)"
  }}
}}

Remember: QUALITY over QUANTITY. Extract 3-5 well-structured policies, NOT 15 fragmented ones."""

        return prompt

    def _parse_extraction_result(
        self, content: str, chunk: Union[Dict, Any]
    ) -> Dict[str, Any]:
        """
        Parse extraction result from LLM response.

        Args:
            content: LLM response content
            chunk: Source chunk

        Returns:
            Dictionary with policies and definitions
        """
        try:
            # Parse JSON from response
            content = content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)

            # Convert to SubPolicy objects
            policies = self._convert_to_policies(result.get("policies", []), chunk)
            definitions = result.get("definitions", {})

            return {
                "policies": policies,
                "definitions": definitions
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {"policies": [], "definitions": {}}
        except Exception as e:
            logger.error(f"Error parsing extraction result: {e}")
            return {"policies": [], "definitions": {}}

    def _convert_to_policies(
        self, raw_policies: List[Dict], chunk: Union[Dict, Any]
    ) -> List[SubPolicy]:
        """
        Convert raw policy data to SubPolicy objects.

        Args:
            raw_policies: List of policy dicts from LLM
            chunk: Source chunk

        Returns:
            List of SubPolicy objects
        """
        policies = []
        parse_errors = 0
        
        start_page = _get_chunk_attr(chunk, "start_page", 0)
        section_context = _get_chunk_attr(chunk, "section_context", "")

        for idx, policy_data in enumerate(raw_policies):
            try:
                # Parse conditions
                conditions = []
                for cond_data in policy_data.get("conditions", []):
                    condition = PolicyCondition(
                        condition_id=cond_data.get("condition_id", ""),
                        description=cond_data.get("description", ""),
                        logic_type=cond_data.get("logic_type", "SIMPLE"),
                        source_references=[
                            SourceReference(
                                page_number=cond_data.get("page_number", start_page),
                                section=cond_data.get("section", section_context),
                                quoted_text=cond_data.get("quoted_text", "")[:500],
                            )
                        ],
                        confidence_score=cond_data.get("confidence_score", 0.7),
                    )
                    conditions.append(condition)

                # Create SubPolicy object
                policy_id = policy_data.get("policy_id", f"policy_{len(policies)}")
                policy = SubPolicy(
                    policy_id=policy_id,
                    title=policy_data.get("title", "Untitled Policy"),
                    description=policy_data.get("description", ""),
                    level=policy_data.get("level", 0),
                    conditions=conditions,
                    source_references=[
                        SourceReference(
                            page_number=policy_data.get("page_number", start_page),
                            section=policy_data.get("section", section_context),
                            quoted_text=policy_data.get("quoted_text", "")[:500],
                        )
                    ],
                    parent_id=policy_data.get("parent_id"),
                    children=[],
                    confidence_score=policy_data.get("confidence_score", 0.7),
                )

                policies.append(policy)

            except Exception as e:
                parse_errors += 1
                logger.error(f"Error parsing policy {idx + 1}/{len(raw_policies)}: {e}")
                continue

        if parse_errors > 0:
            logger.warning(f"Failed to parse {parse_errors}/{len(raw_policies)} policies")

        return policies

    def _deduplicate_policy_ids(self, policies: List[SubPolicy]) -> List[SubPolicy]:
        """
        Detect and fix duplicate policy IDs by renaming duplicates.
        Uses heuristics to maintain parent-child relationships.

        Args:
            policies: List of SubPolicy objects (may contain duplicates)

        Returns:
            List of SubPolicy objects with unique IDs
        """
        from collections import defaultdict

        seen_ids = {}
        deduped_policies = []
        id_renames = []  # Track (original_id, new_id, policy_index)

        # First pass: detect duplicates and rename them
        for idx, policy in enumerate(policies):
            original_id = policy.policy_id

            if original_id in seen_ids:
                # Duplicate found - generate unique ID
                counter = seen_ids[original_id]
                seen_ids[original_id] += 1
                new_id = f"{original_id}_dup{counter}"

                logger.warning(
                    f"Duplicate policy ID '{original_id}' found. "
                    f"Policy '{policy.title}' (level {policy.level}) renamed to '{new_id}'"
                )

                # Track the rename with policy index
                id_renames.append((original_id, new_id, idx))
                policy.policy_id = new_id

            else:
                seen_ids[original_id] = 1

            deduped_policies.append(policy)

        # Second pass: update parent_id references using heuristics
        if id_renames:
            logger.info(f"Analyzing {len(id_renames)} renamed policies for parent_id updates")

            # Group renames by original_id
            renames_by_id = defaultdict(list)
            for original_id, new_id, idx in id_renames:
                renames_by_id[original_id].append((new_id, idx))

            # For each original_id that had duplicates
            for original_id, renamed_list in renames_by_id.items():
                # Find children that reference this original_id
                children_indices = [
                    i for i, p in enumerate(deduped_policies)
                    if p.parent_id == original_id
                ]

                if not children_indices:
                    continue  # No children reference this ID

                # Heuristic: If there's only ONE renamed version, update all children to it
                # If there are multiple renamed versions, keep children pointing to original
                # (let them attach to the first policy that kept the original ID)
                if len(renamed_list) == 1:
                    new_id, renamed_idx = renamed_list[0]
                    logger.info(
                        f"Updating {len(children_indices)} children: "
                        f"parent_id '{original_id}' -> '{new_id}'"
                    )
                    for child_idx in children_indices:
                        deduped_policies[child_idx].parent_id = new_id
                else:
                    logger.warning(
                        f"Multiple policies renamed from '{original_id}'. "
                        f"Keeping {len(children_indices)} children attached to original ID. "
                        f"Manual review may be needed."
                    )

        return deduped_policies

    def _filter_low_quality_policies(self, policies: List[SubPolicy]) -> List[SubPolicy]:
        """
        Filter out low-quality policies that are likely noise or over-extraction.
        
        Criteria for removal:
        - Very short descriptions (<20 chars)
        - No conditions and no children
        - Very generic titles
        - Confidence score <0.4
        - Duplicate semantic content
        
        Args:
            policies: List of SubPolicy objects
        
        Returns:
            Filtered list of high-quality policies
        """
        if not policies:
            return policies
        
        filtered = []
        removed_count = 0
        
        # Track seen content for semantic deduplication
        seen_descriptions = {}
        
        for policy in policies:
            # Rule 1: Must have substantial description
            if len(policy.description.strip()) < 20:
                logger.debug(
                    f"Removing policy '{policy.policy_id}' - description too short ({len(policy.description)} chars)"
                )
                removed_count += 1
                continue
            
            # Rule 2: Leaf policies must have conditions (unless explicitly marked as aggregator)
            if not policy.conditions and not policy.children and policy.level > 0:
                logger.debug(
                    f"Removing policy '{policy.policy_id}' - leaf policy with no conditions"
                )
                removed_count += 1
                continue
            
            # Rule 3: Must have reasonable confidence
            if policy.confidence_score < 0.4:
                logger.debug(
                    f"Removing policy '{policy.policy_id}' - low confidence ({policy.confidence_score:.2f})"
                )
                removed_count += 1
                continue
            
            # Rule 4: Check for generic/meaningless titles
            generic_titles = [
                "policy", "section", "criteria", "requirement", "guideline",
                "untitled", "unnamed", "additional", "other", "general"
            ]
            title_lower = policy.title.lower()
            if any(gen in title_lower for gen in generic_titles) and len(title_lower) < 25:
                # Allow if it has substantial content (multiple conditions)
                if len(policy.conditions) < 2:
                    logger.debug(
                        f"Removing policy '{policy.policy_id}' - generic title with insufficient content"
                    )
                    removed_count += 1
                    continue
            
            # Rule 5: Semantic deduplication - check if description is very similar to existing
            desc_key = policy.description[:100].lower().strip()
            if desc_key in seen_descriptions:
                existing_id = seen_descriptions[desc_key]
                logger.warning(
                    f"Removing policy '{policy.policy_id}' - duplicate description "
                    f"(matches existing '{existing_id}')"
                )
                removed_count += 1
                continue
            
            seen_descriptions[desc_key] = policy.policy_id
            filtered.append(policy)
        
        if removed_count > 0:
            logger.info(
                f"Quality filter: Removed {removed_count}/{len(policies)} low-quality policies "
                f"({removed_count/len(policies)*100:.1f}%)"
            )
        
        return filtered

    def _merge_similar_policies(self, policies: List[SubPolicy]) -> List[SubPolicy]:
        """
        Merge policies that are semantically similar or redundant.
        This reduces over-extraction by combining fragment policies.
        
        Args:
            policies: List of SubPolicy objects
        
        Returns:
            Merged list of policies
        """
        if len(policies) <= 1:
            return policies
        
        from collections import defaultdict
        
        # FIRST PASS: Global duplicate check (same OR similar titles, regardless of parent)
        # This catches true duplicates that may have been assigned different parents
        # Uses semantic similarity, not just exact matching
        
        policies_after_global_merge = []
        merged_ids = set()
        global_merge_count = 0
        
        for i, policy in enumerate(policies):
            if policy.policy_id in merged_ids:
                continue
            
            # Find all policies with similar titles (semantic matching)
            similar_policies = [policy]
            
            for j in range(i + 1, len(policies)):
                if policies[j].policy_id in merged_ids:
                    continue
                
                # Check if titles are semantically similar
                if self._are_titles_semantically_similar(policy.title, policies[j].title):
                    similar_policies.append(policies[j])
                    merged_ids.add(policies[j].policy_id)
            
            # Merge if we found duplicates
            if len(similar_policies) > 1:
                logger.info(
                    f"Found {len(similar_policies)} semantically similar policies: "
                    f"'{similar_policies[0].title}' - IDs: {[p.policy_id for p in similar_policies]}"
                )
                merged_policy = self._merge_policy_group(similar_policies)
                policies_after_global_merge.append(merged_policy)
                global_merge_count += len(similar_policies) - 1
            else:
                policies_after_global_merge.append(policy)
            
            merged_ids.add(policy.policy_id)
        
        if global_merge_count > 0:
            logger.info(
                f"Global semantic merge: {global_merge_count} policies merged, "
                f"reduced from {len(policies)} to {len(policies_after_global_merge)}"
            )
            policies = policies_after_global_merge
        
        # SECOND PASS: Group-based similarity merge (within same parent/level)
        merged = []
        merged_ids = set()
        merge_count = 0
        
        # Group policies by parent_id and level
        groups = defaultdict(list)
        
        for policy in policies:
            key = (policy.parent_id, policy.level)
            groups[key].append(policy)
        
        # Process each group
        for (parent_id, level), group_policies in groups.items():
            if len(group_policies) == 1:
                merged.extend(group_policies)
                continue
            
            # Check for mergeable policies in this group
            i = 0
            while i < len(group_policies):
                if group_policies[i].policy_id in merged_ids:
                    i += 1
                    continue
                
                current = group_policies[i]
                candidates_to_merge = [current]
                
                # Look for similar policies to merge
                for j in range(i + 1, len(group_policies)):
                    if group_policies[j].policy_id in merged_ids:
                        continue
                    
                    candidate = group_policies[j]
                    
                    # Check if titles are similar (simple heuristic)
                    if self._are_titles_similar(current.title, candidate.title):
                        candidates_to_merge.append(candidate)
                        merged_ids.add(candidate.policy_id)
                
                # If we found similar policies, merge them
                if len(candidates_to_merge) > 1:
                    logger.info(
                        f"Merging {len(candidates_to_merge)} similar policies: "
                        f"{[p.policy_id for p in candidates_to_merge]}"
                    )
                    merged_policy = self._merge_policy_group(candidates_to_merge)
                    merged.append(merged_policy)
                    merge_count += len(candidates_to_merge) - 1
                else:
                    merged.append(current)
                
                merged_ids.add(current.policy_id)
                i += 1
        
        if merge_count > 0:
            logger.info(
                f"Merged {merge_count} policies, "
                f"reduced from {len(policies)} to {len(merged)}"
            )
        
        return merged
    
    def _are_titles_similar(self, title1: str, title2: str) -> bool:
        """Check if two policy titles are semantically similar."""
        # Simple heuristic: check word overlap
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = words1 & words2
        min_length = min(len(words1), len(words2))
        
        # If >70% of words overlap, consider similar
        return len(overlap) / min_length > 0.7
    
    def _are_titles_semantically_similar(self, title1: str, title2: str) -> bool:
        """
        Semantic similarity check for policy titles.
        Uses multiple signals to detect duplicates even if wording differs slightly.
        """
        # Normalize titles
        t1 = title1.lower().strip()
        t2 = title2.lower().strip()
        
        # Exact match
        if t1 == t2:
            return True
        
        # Remove common prefixes/suffixes that might vary
        prefixes = ['criteria for', 'guidelines for', 'policy for', 'requirements for']
        suffixes = ['criteria', 'guidelines', 'policy', 'requirements', 'procedures']
        
        for prefix in prefixes:
            t1 = t1.replace(prefix, '').strip()
            t2 = t2.replace(prefix, '').strip()
        
        # Extract key words (remove stop words)
        stop_words = {'for', 'of', 'the', 'and', 'or', 'in', 'a', 'an', 'to', 'is', 'are', 'be'}
        words1 = set(w for w in t1.split() if w not in stop_words and len(w) > 2)
        words2 = set(w for w in t2.split() if w not in stop_words and len(w) > 2)
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity
        intersection = words1 & words2
        union = words1 | words2
        jaccard_sim = len(intersection) / len(union) if union else 0
        
        # Check for key phrase matches (order matters)
        key_phrases_1 = self._extract_key_phrases(t1)
        key_phrases_2 = self._extract_key_phrases(t2)
        
        phrase_overlap = len(key_phrases_1 & key_phrases_2) / max(len(key_phrases_1), len(key_phrases_2)) if key_phrases_1 or key_phrases_2 else 0
        
        # Consider similar if:
        # 1. Jaccard similarity > 0.60 (60% word overlap - aggressive to catch "medically/medical"), OR
        # 2. Key phrases overlap > 0.65 (65% phrase overlap - more lenient), OR
        # 3. Root word overlap > 0.70 (70% root word match - more lenient)
        if jaccard_sim > 0.60:
            return True
        
        if phrase_overlap > 0.65:
            return True
        
        # Special case: Check for common patterns indicating duplicates
        # Example: "Medically Necessary Criteria" vs "Medical Necessity Criteria"
        # Extract root words (stem-like matching)
        roots1 = set(self._get_root_word(w) for w in words1)
        roots2 = set(self._get_root_word(w) for w in words2)
        
        root_overlap = len(roots1 & roots2) / min(len(roots1), len(roots2)) if roots1 and roots2 else 0
        
        # If >70% of root words match, likely duplicate (lowered from 80% for better recall)
        if root_overlap > 0.70:
            return True
        
        return False
    
    def _extract_key_phrases(self, text: str) -> set:
        """Extract key 2-3 word phrases from text."""
        words = text.split()
        phrases = set()
        
        # Extract bigrams
        for i in range(len(words) - 1):
            phrases.add(f"{words[i]} {words[i+1]}")
        
        # Extract trigrams
        for i in range(len(words) - 2):
            phrases.add(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        return phrases
    
    def _get_root_word(self, word: str) -> str:
        """Simple root word extraction (pseudo-stemming)."""
        # Remove common suffixes
        suffixes = ['tion', 'ness', 'ment', 'ity', 'ies', 'ing', 'ed', 'er', 'est', 'ly', 's']
        
        for suffix in sorted(suffixes, key=len, reverse=True):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        
        return word
    
    def _merge_policy_group(self, policies: List[SubPolicy]) -> SubPolicy:
        """Merge multiple similar policies into one comprehensive policy."""
        if len(policies) == 1:
            return policies[0]
        
        # Use the first policy as base
        base = policies[0]
        
        # Combine conditions from all policies
        all_conditions = []
        condition_ids = set()
        
        for policy in policies:
            for cond in policy.conditions:
                if cond.condition_id not in condition_ids:
                    all_conditions.append(cond)
                    condition_ids.add(cond.condition_id)
        
        # Combine source references
        all_refs = []
        ref_keys = set()
        
        for policy in policies:
            for ref in policy.source_references:
                key = (ref.page_number, ref.quoted_text[:50])
                if key not in ref_keys:
                    all_refs.append(ref)
                    ref_keys.add(key)
        
        # Use longest description
        longest_desc = max(policies, key=lambda p: len(p.description)).description
        
        # Average confidence
        avg_confidence = sum(p.confidence_score for p in policies) / len(policies)
        
        # Create merged policy
        merged = SubPolicy(
            policy_id=base.policy_id,
            title=base.title,
            description=longest_desc,
            level=base.level,
            conditions=all_conditions,
            source_references=all_refs,
            parent_id=base.parent_id,
            children=[],  # Children will be linked later
            confidence_score=avg_confidence,
        )
        
        return merged

    def _build_hierarchy(
        self, policies: List[SubPolicy], definitions: Dict[str, str]
    ) -> PolicyHierarchy:
        """
        Build hierarchical structure from flat list of policies.

        Args:
            policies: List of SubPolicy objects
            definitions: Dictionary of definitions

        Returns:
            PolicyHierarchy object
        """
        logger.debug(f"Building hierarchy from {len(policies)} policies...")

        # Step 1: Detect and fix duplicate policy IDs
        logger.debug("Step 1: Checking for duplicate policy IDs...")
        policies = self._deduplicate_policy_ids(policies)
        
        # Step 2: Filter low-quality policies
        logger.debug("Step 2: Filtering low-quality policies...")
        policies = self._filter_low_quality_policies(policies)
        
        # Step 3: Merge similar policies to reduce fragmentation
        logger.debug("Step 3: Merging similar policies...")
        policies = self._merge_similar_policies(policies)

        # Step 4: Create policy lookup by ID
        logger.debug("Step 4: Creating policy lookup map...")
        policy_map = {p.policy_id: p for p in policies}

        # Step 5: Build parent-child relationships
        logger.debug("Step 5: Building parent-child relationships...")
        root_policies = []
        child_count = 0

        for policy in policies:
            if policy.parent_id and policy.parent_id in policy_map:
                # Add to parent's children
                parent = policy_map[policy.parent_id]
                parent.children.append(policy)
                child_count += 1
                logger.debug(f"Linked child policy '{policy.policy_id}' to parent '{policy.parent_id}'")
            else:
                # Root policy
                root_policies.append(policy)

        logger.info(f"Hierarchy structure (pre-refinement) - Root policies: {len(root_policies)}, Child policies: {child_count}")
        
        # Step 6: Refine hierarchy using semantic analysis
        logger.debug("Step 6: Refining hierarchy with semantic analysis...")
        root_policies, policies = self._refine_policy_hierarchy(root_policies, policies, policy_map)
        
        # Step 7: Fix orphaned policies (policies with parent_id but parent doesn't exist)
        logger.debug("Step 7: Fixing orphaned policies...")
        orphaned_fixed = self._fix_orphaned_policies(policies, policy_map, root_policies)
        if orphaned_fixed > 0:
            logger.info(f"Fixed {orphaned_fixed} orphaned policies by promoting them to root policies")

        # Step 8: Fix levels and ensure consistency
        logger.debug("Step 8: Fixing policy levels and ensuring consistency...")
        self._fix_policy_levels(policies)

        # Recalculate child count after fixing orphans
        child_count = sum(1 for p in policies if p.parent_id)
        logger.info(f"Hierarchy structure (post-refinement) - Root policies: {len(root_policies)}, Child policies: {child_count}")

        # Step 9: Log detailed hierarchy structure
        self._log_hierarchy_structure(root_policies, policies)

        # Calculate max depth
        logger.debug("Calculating hierarchy depth...")
        max_depth = self._calculate_max_depth(root_policies)

        # Sort root policies by ID for consistency
        logger.debug("Sorting root policies...")
        root_policies.sort(key=lambda p: p.policy_id)

        hierarchy = PolicyHierarchy(
            root_policies=root_policies,
            total_policies=len(policies),
            max_depth=max_depth,
            definitions=definitions,
        )

        logger.info(f"Hierarchy built successfully - Total: {len(policies)}, Roots: {len(root_policies)}, Max depth: {max_depth}")

        return hierarchy
    
    def _refine_policy_hierarchy(
        self, 
        root_policies: List[SubPolicy], 
        all_policies: List[SubPolicy],
        policy_map: Dict[str, SubPolicy]
    ) -> tuple[List[SubPolicy], List[SubPolicy]]:
        """
        Refine policy hierarchy using semantic analysis to group related policies.
        
        This addresses the "too many roots" problem by:
        1. Detecting umbrella policies (broad, high-level policies)
        2. Finding best parents for orphaned root policies
        3. Re-parenting policies based on keyword and semantic similarity
        
        Args:
            root_policies: List of root policies
            all_policies: List of all policies
            policy_map: Dictionary mapping policy IDs to policies
        
        Returns:
            Tuple of (refined_root_policies, all_policies)
        """
        logger.info(f"Starting semantic hierarchy refinement with {len(root_policies)} root policies...")
        
        # STEP 1: Identify umbrella policies (candidates to be parents)
        umbrella_policies = self._identify_umbrella_policies(root_policies)
        logger.info(f"Identified {len(umbrella_policies)} umbrella policies: {[p.policy_id for p in umbrella_policies]}")
        
        # STEP 2: Find policies that should be children
        candidate_children = [p for p in root_policies if p not in umbrella_policies]
        logger.info(f"Found {len(candidate_children)} candidate children for re-parenting")
        
        # STEP 3: Match children to best parents
        reparented_count = 0
        for child_candidate in candidate_children:
            best_parent = self._find_best_parent(child_candidate, umbrella_policies, all_policies)
            
            if best_parent:
                # Re-parent the policy
                logger.info(
                    f"Re-parenting '{child_candidate.policy_id}' ({child_candidate.title[:50]}) "
                    f"under '{best_parent.policy_id}' ({best_parent.title[:50]})"
                )
                
                child_candidate.parent_id = best_parent.policy_id
                child_candidate.level = best_parent.level + 1
                
                # Add to parent's children
                if child_candidate not in best_parent.children:
                    best_parent.children.append(child_candidate)
                
                reparented_count += 1
        
        # STEP 4: Rebuild root list (only policies without parents)
        refined_roots = [p for p in all_policies if not p.parent_id]
        
        logger.info(
            f"Semantic refinement complete: {reparented_count} policies re-parented, "
            f"root count reduced from {len(root_policies)} to {len(refined_roots)}"
        )
        
        return refined_roots, all_policies
    
    def _identify_umbrella_policies(self, policies: List[SubPolicy]) -> List[SubPolicy]:
        """
        Identify umbrella policies that should serve as parents.
        
        Umbrella policies are broad, high-level policies that logically group others.
        Indicators:
        - Title contains keywords like "criteria", "requirements", "eligibility"
        - Longer description (more comprehensive)
        - More conditions (more detailed)
        - Comes earlier in document (foundation policies)
        
        Args:
            policies: List of policies to analyze
        
        Returns:
            List of umbrella policies
        """
        umbrella_indicators = []
        
        for policy in policies:
            score = 0
            reasons = []
            
            title_lower = policy.title.lower()
            desc_lower = policy.description.lower()
            
            # INDICATOR 1: Title contains umbrella keywords
            umbrella_keywords = ['criteria', 'requirements', 'eligibility', 'medically necessary', 'coverage']
            if any(kw in title_lower for kw in umbrella_keywords):
                score += 2
                reasons.append("umbrella_keyword_in_title")
            
            # INDICATOR 2: Longer description (comprehensive policy)
            if len(policy.description.split()) > 25:
                score += 1
                reasons.append("comprehensive_description")
            
            # INDICATOR 3: Multiple conditions (detailed policy)
            if len(policy.conditions) >= 3:
                score += 1
                reasons.append("multiple_conditions")
            
            # INDICATOR 4: Already has children (LLM identified it as parent)
            if policy.children:
                score += 2
                reasons.append("has_existing_children")
            
            # INDICATOR 5: Early in document (foundation policy)
            if hasattr(policy, 'source_references') and policy.source_references:
                first_page = min(ref.page_number for ref in policy.source_references)
                if first_page <= 3:
                    score += 1
                    reasons.append("early_in_document")
            
            umbrella_indicators.append((policy, score, reasons))
        
        # Select policies with score >= 2 as umbrella policies (lowered from 3 for more aggressive grouping)
        umbrellas = [p for p, score, reasons in umbrella_indicators if score >= 2]
        
        # Log umbrella detection details
        for policy, score, reasons in umbrella_indicators:
            if score >= 2:
                logger.debug(f"Umbrella policy: '{policy.policy_id}' (score={score}, reasons={reasons})")
        
        # If no umbrellas found, select top 2-3 by score
        if not umbrellas:
            sorted_by_score = sorted(umbrella_indicators, key=lambda x: x[1], reverse=True)
            umbrellas = [p for p, _, _ in sorted_by_score[:min(3, len(sorted_by_score))]]
            logger.info(f"No clear umbrellas, selecting top {len(umbrellas)} by score")
        
        return umbrellas
    
    def _find_best_parent(
        self, 
        child: SubPolicy, 
        parent_candidates: List[SubPolicy],
        all_policies: List[SubPolicy]
    ) -> SubPolicy:
        """
        Find the best parent for a child policy using multi-signal matching.
        
        Signals:
        1. Keyword overlap (common terms indicate related content)
        2. Page proximity (nearby policies are often related)
        3. Title similarity (similar naming suggests relationship)
        
        Args:
            child: Policy to find parent for
            parent_candidates: List of potential parents
            all_policies: All policies (for context)
        
        Returns:
            Best parent policy, or None if no good match
        """
        if not parent_candidates:
            return None
        
        scores = []
        
        for parent in parent_candidates:
            score = 0.0
            signals = {}
            
            # SIGNAL 1: Keyword Overlap (50% weight - increased from 40% for stronger semantic matching)
            child_keywords = set(self._extract_keywords(child.title + ' ' + child.description))
            parent_keywords = set(self._extract_keywords(parent.title + ' ' + parent.description))
            
            if child_keywords and parent_keywords:
                keyword_overlap = len(child_keywords & parent_keywords) / len(child_keywords)
                
                # Boost score if overlapping keywords include domain-specific terms
                domain_keywords = {'bariatric', 'surgery', 'adolescent', 'criteria', 'bmi', 'obesity'}
                overlapping_domain_terms = (child_keywords & parent_keywords) & domain_keywords
                if overlapping_domain_terms:
                    keyword_overlap *= 1.2  # 20% boost for domain-specific overlap
                
                score += keyword_overlap * 0.5
                signals['keyword_overlap'] = keyword_overlap
            
            # SIGNAL 2: Page Proximity (15% weight - decreased from 20%)
            if hasattr(child, 'source_references') and hasattr(parent, 'source_references'):
                if child.source_references and parent.source_references:
                    child_first_page = min(ref.page_number for ref in child.source_references)
                    parent_first_page = min(ref.page_number for ref in parent.source_references)
                    page_distance = abs(child_first_page - parent_first_page)
                    
                    # Closer pages = higher score
                    proximity_score = 1.0 / (1.0 + page_distance)
                    score += proximity_score * 0.15
                    signals['page_proximity'] = proximity_score
            
            # SIGNAL 3: Title Pattern Matching (20% weight)
            # E.g., "Bariatric Procedure Guidelines" should be child of policy containing "Bariatric"
            parent_title_words = set(parent.title.lower().split())
            child_title_words = set(child.title.lower().split())
            
            # Check if parent title is subset of child (or vice versa)
            if parent_title_words & child_title_words:
                title_overlap = len(parent_title_words & child_title_words) / max(len(parent_title_words), len(child_title_words))
                score += title_overlap * 0.2
                signals['title_overlap'] = title_overlap
            
            # SIGNAL 4: Condition Similarity (15% weight - decreased from 20%)
            # Extract condition descriptions from PolicyCondition objects
            child_conditions_text = ' '.join([
                c.description if hasattr(c, 'description') else str(c) 
                for c in child.conditions
            ])
            parent_conditions_text = ' '.join([
                c.description if hasattr(c, 'description') else str(c) 
                for c in parent.conditions
            ])
            
            child_cond_keywords = set(self._extract_keywords(child_conditions_text))
            parent_cond_keywords = set(self._extract_keywords(parent_conditions_text))
            
            if child_cond_keywords and parent_cond_keywords:
                cond_overlap = len(child_cond_keywords & parent_cond_keywords) / len(child_cond_keywords)
                score += cond_overlap * 0.15
                signals['condition_overlap'] = cond_overlap
            
            scores.append((parent, score, signals))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return best parent if score > threshold (0.20 - lowered from 0.25 for deeper hierarchy)
        # This creates more parent-child relationships, resulting in deeper nesting and grandchildren
        # Lower threshold = more aggressive parent matching = fewer orphan roots
        if scores[0][1] > 0.20:
            best_parent, best_score, best_signals = scores[0]
            logger.debug(
                f"Best parent for '{child.policy_id}': '{best_parent.policy_id}' "
                f"(score={best_score:.2f}, signals={best_signals})"
            )
            return best_parent
        
        logger.debug(f"No suitable parent found for '{child.policy_id}' (best score={scores[0][1]:.2f} < threshold 0.20)")
        return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from text for semantic matching.
        
        Args:
            text: Text to extract keywords from
        
        Returns:
            List of keywords
        """
        if not text:
            return []
        
        # Simple keyword extraction (can be improved with NLP)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their'
        }
        
        # Keep only meaningful words (length > 3, not stop words)
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        
        return keywords

    def _fix_orphaned_policies(
        self,
        policies: List[SubPolicy],
        policy_map: Dict[str, SubPolicy],
        root_policies: List[SubPolicy]
    ) -> int:
        """
        Fix orphaned policies (policies with parent_id but parent doesn't exist).

        This happens when:
        1. Policies are merged/deduplicated and their IDs change
        2. Parent policies are deleted during merging
        3. Semantic refinement creates parent_id references that don't exist

        Solution: Try to find an alternative parent, or promote to root as last resort.

        Args:
            policies: List of all policies
            policy_map: Dictionary mapping policy IDs to policies
            root_policies: List of root policies (will be modified)

        Returns:
            Number of orphaned policies fixed
        """
        orphaned_count = 0

        for policy in policies:
            # Check if policy has a parent_id but parent doesn't exist
            if policy.parent_id and policy.parent_id not in policy_map:
                logger.warning(
                    f"Orphaned policy detected: '{policy.policy_id}' ({policy.title}) "
                    f"has parent_id='{policy.parent_id}' but parent not found."
                )

                # Try to find an alternative parent based on semantic similarity
                alternative_parent = self._find_alternative_parent(policy, list(policy_map.values()))

                if alternative_parent:
                    logger.info(
                        f"Found alternative parent for '{policy.policy_id}': "
                        f"'{alternative_parent.policy_id}' ({alternative_parent.title})"
                    )
                    policy.parent_id = alternative_parent.policy_id
                    policy.level = alternative_parent.level + 1

                    # Add to parent's children
                    if policy not in alternative_parent.children:
                        alternative_parent.children.append(policy)
                else:
                    # No alternative found, promote to root
                    logger.warning(
                        f"No alternative parent found for '{policy.policy_id}'. "
                        f"Promoting to root policy."
                    )
                    policy.parent_id = None
                    policy.level = 0

                    # Add to root policies if not already there
                    if policy not in root_policies:
                        root_policies.append(policy)

                orphaned_count += 1

        return orphaned_count

    def _find_alternative_parent(
        self,
        orphan: SubPolicy,
        all_policies: List[SubPolicy]
    ) -> Optional[SubPolicy]:
        """
        Find an alternative parent for an orphaned policy using semantic matching.

        Args:
            orphan: Orphaned policy needing a parent
            all_policies: All available policies

        Returns:
            Best matching parent policy or None
        """
        # Only consider policies that could be parents (not already children)
        parent_candidates = [
            p for p in all_policies
            if p.policy_id != orphan.policy_id and not p.parent_id
        ]

        if not parent_candidates:
            return None

        # Score each candidate based on similarity
        scored_candidates = []

        for candidate in parent_candidates:
            score = 0
            reasons = []

            # Factor 1: Keyword overlap in titles
            orphan_keywords = set(self._extract_keywords(orphan.title))
            candidate_keywords = set(self._extract_keywords(candidate.title))
            keyword_overlap = len(orphan_keywords & candidate_keywords)
            if keyword_overlap > 0:
                score += keyword_overlap * 2
                reasons.append(f"keyword_overlap={keyword_overlap}")

            # Factor 2: Check if candidate is umbrella policy
            candidate_title_lower = candidate.title.lower()
            if any(kw in candidate_title_lower for kw in ['criteria', 'eligibility', 'requirements', 'medically necessary']):
                score += 3
                reasons.append("umbrella_policy")

            # Factor 3: Candidate has more conditions (more comprehensive)
            if len(candidate.conditions) > len(orphan.conditions):
                score += 1
                reasons.append("more_comprehensive")

            # Factor 4: Page proximity
            if hasattr(orphan, 'source_references') and hasattr(candidate, 'source_references'):
                if orphan.source_references and candidate.source_references:
                    orphan_page = orphan.source_references[0].page_number
                    candidate_page = candidate.source_references[0].page_number
                    if abs(orphan_page - candidate_page) <= 2:
                        score += 1
                        reasons.append("page_proximity")

            if score > 0:
                scored_candidates.append((candidate, score, reasons))

        # Return best candidate (score >= 3)
        if scored_candidates:
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            best_candidate, best_score, reasons = scored_candidates[0]

            if best_score >= 3:
                logger.debug(
                    f"Alternative parent match: '{best_candidate.policy_id}' "
                    f"(score={best_score}, reasons={reasons})"
                )
                return best_candidate

        return None

    def _fix_policy_levels(self, policies: List[SubPolicy]) -> None:
        """
        Fix policy level assignments to ensure consistency.
        
        Ensures that:
        - Root policies have level = 0
        - Child policies have level = parent.level + 1
        - No inconsistent level assignments
        
        Args:
            policies: List of all policies
        """
        policy_map = {p.policy_id: p for p in policies}
        
        # First pass: Set all root policies to level 0
        for policy in policies:
            if not policy.parent_id:
                if policy.level != 0:
                    logger.debug(f"Fixing root policy '{policy.policy_id}' level: {policy.level} → 0")
                    policy.level = 0
        
        # Second pass: Set child policies based on parent level
        for policy in policies:
            if policy.parent_id and policy.parent_id in policy_map:
                parent = policy_map[policy.parent_id]
                expected_level = parent.level + 1
                
                if policy.level != expected_level:
                    logger.debug(
                        f"Fixing child policy '{policy.policy_id}' level: {policy.level} → {expected_level} "
                        f"(parent '{parent.policy_id}' is level {parent.level})"
                    )
                    policy.level = expected_level
    
    def _log_hierarchy_structure(self, root_policies: List[SubPolicy], all_policies: List[SubPolicy]) -> None:
        """
        Log detailed hierarchy structure for debugging.
        
        Args:
            root_policies: List of root policies
            all_policies: List of all policies
        """
        logger.info("=" * 80)
        logger.info("POLICY HIERARCHY STRUCTURE")
        logger.info("=" * 80)
        
        for root in root_policies:
            self._log_policy_tree(root, indent=0)
        
        # Log orphaned policies (have parent_id but parent not found)
        orphans = [p for p in all_policies if p.parent_id and p not in root_policies and not any(p in r.children for r in root_policies)]
        if orphans:
            logger.warning(f"Found {len(orphans)} orphaned policies (parent not found):")
            for orphan in orphans:
                logger.warning(f"  - {orphan.policy_id} (parent_id={orphan.parent_id})")
        
        logger.info("=" * 80)
    
    def _log_policy_tree(self, policy: SubPolicy, indent: int = 0) -> None:
        """
        Recursively log policy tree structure.
        
        Args:
            policy: Policy to log
            indent: Indentation level
        """
        prefix = "  " * indent
        logger.info(
            f"{prefix}[L{policy.level}] {policy.policy_id}: {policy.title[:60]} "
            f"({len(policy.conditions)} conditions, {len(policy.children)} children)"
        )
        
        for child in policy.children:
            self._log_policy_tree(child, indent + 1)

    def _calculate_max_depth(self, policies: List[SubPolicy], current_depth: int = 0) -> int:
        """
        Calculate maximum depth of policy hierarchy.

        Args:
            policies: List of SubPolicy objects
            current_depth: Current depth level

        Returns:
            Maximum depth
        """
        if not policies:
            return current_depth

        max_child_depth = current_depth

        for policy in policies:
            if policy.children:
                child_depth = self._calculate_max_depth(policy.children, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth

    async def extract_additional_context(
        self, policy: SubPolicy, pages: List[PDFPage]
    ) -> Dict[str, Any]:
        """
        Extract additional context for a specific policy.

        Args:
            policy: SubPolicy object
            pages: List of PDFPage objects

        Returns:
            Additional context dictionary
        """
        # Get relevant pages
        page_numbers = [ref.page_number for ref in policy.source_references]
        relevant_text = ""

        for page in pages:
            if page.page_number in page_numbers:
                relevant_text += f"\n\n--- Page {page.page_number} ---\n{page.text}"

        if not relevant_text:
            return {}

        prompt = f"""Analyze this policy and provide additional context:

Policy: {policy.title}
Description: {policy.description}

Source Text:
{relevant_text[:3000]}

Provide:
1. Key stakeholders affected
2. Related policies or dependencies
3. Exceptions or special cases
4. Potential ambiguities

Return as JSON."""

        try:
            if self.llm is None:
                logger.warning("LLM not initialized for additional context extraction")
                return {}

            response = await self.llm.ainvoke(prompt)
            
            content = response.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            return json.loads(content)

        except Exception as e:
            logger.error(f"Error extracting additional context: {e}")
            return {}
