"""
Policy extraction engine with hierarchy detection.
"""
import json
import re
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import settings
from app.utils.logger import get_logger
from app.core.chunking_strategy import DocumentChunk
from app.core.pdf_processor import PDFPage
from app.models.schemas import (
    SubPolicy,
    PolicyCondition,
    PolicyHierarchy,
    SourceReference,
)

logger = get_logger(__name__)


class PolicyExtractor:
    """Extracts policies and sub-policies from documents with hierarchy detection."""

    def __init__(self, use_gpt4: bool = False):
        """
        Initialize policy extractor.

        Args:
            use_gpt4: Whether to use GPT-4 for extraction
        """
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model_complex if use_gpt4 else settings.openai_model_primary
        self.use_gpt4 = use_gpt4
        logger.info(f"Policy extractor initialized with model: {self.model} (GPT-4: {use_gpt4})")
        logger.debug(f"Configuration - Max concurrent: {settings.openai_max_concurrent_requests}, Request timeout: {settings.openai_per_request_timeout}s, Max retries: {settings.openai_max_retries}")

    async def extract_policies(
        self, chunks: List[DocumentChunk], pages: List[PDFPage]
    ) -> PolicyHierarchy:
        """
        Extract all policies and their hierarchy from document chunks in parallel.

        Args:
            chunks: List of DocumentChunk objects
            pages: List of PDFPage objects for reference

        Returns:
            PolicyHierarchy object
        """
        import asyncio

        # Create semaphore to limit concurrent API requests
        semaphore = asyncio.Semaphore(settings.openai_max_concurrent_requests)

        logger.info(f"Starting parallel policy extraction from {len(chunks)} chunks")
        logger.info(f"Using model: {self.model} with max {settings.openai_max_concurrent_requests} concurrent requests")
        logger.debug(f"Chunk statistics - Total: {len(chunks)}, Pages covered: {chunks[0].start_page if chunks else 0}-{chunks[-1].end_page if chunks else 0}")

        async def extract_with_semaphore(chunk: DocumentChunk, pages: List[PDFPage], index: int, total: int):
            async with semaphore:
                return await self._extract_chunk_with_timeout(chunk, pages, index, total)

        # Create tasks for all chunks with timeout wrapper
        tasks = []
        for i, chunk in enumerate(chunks):
            tasks.append(extract_with_semaphore(chunk, pages, i + 1, len(chunks)))

        logger.info(f"Executing {len(tasks)} extraction tasks in parallel...")
        # Run all tasks with concurrency control
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all policies and definitions
        all_policies = []
        definitions = {}
        errors_count = 0
        empty_count = 0

        logger.info("Processing extraction results...")
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors_count += 1
                logger.error(f"Error extracting from chunk {i + 1}: {result}")
                continue

            if result is None:
                empty_count += 1
                logger.warning(f"Chunk {i + 1} returned no results")
                continue

            chunk_policies, chunk_definitions = result
            all_policies.extend(chunk_policies)
            definitions.update(chunk_definitions)
            logger.debug(f"Chunk {i + 1} contributed {len(chunk_policies)} policies and {len(chunk_definitions)} definitions")

        logger.info(f"Extraction complete - Collected {len(all_policies)} policies and {len(definitions)} definitions from {len(chunks) - errors_count - empty_count}/{len(chunks)} successful chunks")
        if errors_count > 0:
            logger.warning(f"Failed chunks: {errors_count}")
        if empty_count > 0:
            logger.warning(f"Empty chunks: {empty_count}")

        # Build hierarchy
        logger.info("Building policy hierarchy from extracted policies...")
        policy_hierarchy = self._build_hierarchy(all_policies, definitions)

        logger.info(f"Policy extraction complete - Total policies: {policy_hierarchy.total_policies}, Root policies: {len(policy_hierarchy.root_policies)}, Max depth: {policy_hierarchy.max_depth}, Definitions: {len(definitions)}")

        return policy_hierarchy

    async def _extract_chunk_with_timeout(
        self, chunk: DocumentChunk, pages: List[PDFPage], index: int, total: int
    ) -> tuple[List[SubPolicy], Dict[str, str]]:
        """
        Extract from chunk with timeout.

        Args:
            chunk: DocumentChunk object
            pages: List of PDFPage objects
            index: Current chunk index
            total: Total number of chunks

        Returns:
            Tuple of (policies, definitions) or None on timeout
        """
        import asyncio

        try:
            logger.debug(f"[Chunk {index}/{total}] Starting extraction (Pages {chunk.start_page}-{chunk.end_page}, {len(chunk.text)} chars)")

            result = await asyncio.wait_for(
                self._extract_from_chunk(chunk, pages),
                timeout=settings.openai_per_request_timeout
            )

            policies, defs = result
            logger.info(f"[Chunk {index}/{total}] Extraction complete - {len(policies)} policies, {len(defs)} definitions extracted")
            return result

        except asyncio.TimeoutError:
            logger.error(
                f"[Chunk {index}/{total}] Timeout after {settings.openai_per_request_timeout}s "
                f"(Pages {chunk.start_page}-{chunk.end_page})"
            )
            return [], {}

        except Exception as e:
            logger.error(f"[Chunk {index}/{total}] Error during extraction: {e}", exc_info=True)
            return [], {}

    @retry(
        stop=stop_after_attempt(settings.openai_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _extract_from_chunk(
        self, chunk: DocumentChunk, pages: List[PDFPage]
    ) -> tuple[List[SubPolicy], Dict[str, str]]:
        """
        Extract policies from a single chunk.

        Args:
            chunk: DocumentChunk object
            pages: List of PDFPage objects

        Returns:
            Tuple of (list of SubPolicy objects, definitions dict)
        """
        prompt = self._create_extraction_prompt(chunk)
        prompt_length = len(prompt)

        try:
            logger.debug(f"Calling {self.model} API for chunk {chunk.chunk_id} (prompt: {prompt_length} chars)")

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert policy analyst. Extract policies, sub-policies, and conditions from documents with perfect accuracy. Always provide source references and maintain hierarchical structure.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            logger.debug(f"Received response from {self.model} for chunk {chunk.chunk_id}")

            result = json.loads(response.choices[0].message.content)

            # Parse result into SubPolicy objects
            logger.debug(f"Parsing extraction result for chunk {chunk.chunk_id}...")
            policies = self._parse_extraction_result(result, chunk, pages)
            definitions = result.get("definitions", {})

            logger.debug(f"Parsed {len(policies)} policies and {len(definitions)} definitions from chunk {chunk.chunk_id}")
            return policies, definitions

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for chunk {chunk.chunk_id}: {e}")
            return [], {}
        except Exception as e:
            logger.error(f"Error extracting from chunk {chunk.chunk_id}: {e}")
            return [], {}

    def _create_extraction_prompt(self, chunk: DocumentChunk) -> str:
        """
        Create extraction prompt for a chunk.

        Args:
            chunk: DocumentChunk object

        Returns:
            Prompt string
        """
        prompt = f"""Extract ALL policies, sub-policies, conditions, and definitions from the following document section.

Document Section (Pages {chunk.start_page}-{chunk.end_page}):
{chunk.text}

Instructions:
1. Extract EVERY policy and sub-policy, identifying hierarchical relationships
   - Main policies are typically numbered (1., 2., A., B., etc.)
   - Sub-policies are indented or have sub-numbering (1.1, 1.2, A.i, A.ii, etc.)
   - Look for policies that modify, restrict, or provide exceptions to parent policies

2. For each policy, extract:
   - Unique identifier (use section numbers if available, otherwise generate)
   - Title or heading (extract from document headings)
   - Description/summary
   - Hierarchy level:
     * 0 for top-level main policies
     * 1 for sub-policies under main policies
     * 2 for sub-sub-policies, etc.
   - All conditions that apply to this specific policy
   - Parent policy ID if it's a sub-policy (IMPORTANT: link child policies to their parents)
   - Page number and quoted source text

3. For each condition, extract:
   - Condition ID
   - Description (be specific and detailed)
   - Logic type:
     * "AND" if ALL conditions must be met
     * "OR" if ANY condition can be met
     * "NOT" for exclusions
     * "SIMPLE" for single conditions
   - Source reference with page number and quoted text

4. Extract key definitions and terms from the document

5. Assign confidence scores (0-1) based on:
   - Clarity of the source text (0.9-1.0 for clear, explicit statements)
   - Completeness of information (0.7-0.9 for partially complete)
   - Potential for ambiguity (0.5-0.7 for ambiguous statements)

CRITICAL: Pay special attention to hierarchical structure. If a policy section contains:
- Main heading: create a parent policy (level 0)
- Sub-headings or numbered subsections: create child policies (level 1) with parent_id set
- Further subsections: create grandchild policies (level 2) with appropriate parent_id

Return a JSON object with this structure:
{{
  "policies": [
    {{
      "policy_id": "section_1",
      "title": "Main Policy Title",
      "description": "Comprehensive description",
      "level": 0,
      "parent_id": null,
      "conditions": [
        {{
          "condition_id": "cond_1_1",
          "description": "Specific condition description",
          "logic_type": "AND",
          "page_number": 5,
          "quoted_text": "exact text from source",
          "confidence_score": 0.95
        }}
      ],
      "page_number": 5,
      "section": "Section 1",
      "quoted_text": "exact text from source",
      "confidence_score": 0.9
    }},
    {{
      "policy_id": "section_1_1",
      "title": "Sub-Policy Title",
      "description": "Sub-policy description",
      "level": 1,
      "parent_id": "section_1",
      "conditions": [...],
      "page_number": 6,
      "section": "Section 1.1",
      "quoted_text": "exact text from source",
      "confidence_score": 0.9
    }}
  ],
  "definitions": {{
    "Medical Necessity": "definition text",
    "Bariatric Surgery": "definition text"
  }}
}}

Be thorough and precise. Extract everything, maintaining the hierarchical structure."""

        return prompt

    def _parse_extraction_result(
        self, result: Dict[str, Any], chunk: DocumentChunk, pages: List[PDFPage]
    ) -> List[SubPolicy]:
        """
        Parse extraction result into SubPolicy objects.

        Args:
            result: Extraction result dictionary
            chunk: Source chunk
            pages: List of PDFPage objects

        Returns:
            List of SubPolicy objects
        """
        policies = []
        parse_errors = 0

        raw_policies = result.get("policies", [])
        logger.debug(f"Parsing {len(raw_policies)} policies from LLM response...")

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
                                page_number=cond_data.get("page_number", chunk.start_page),
                                section=cond_data.get("section", chunk.section_context),
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
                            page_number=policy_data.get("page_number", chunk.start_page),
                            section=policy_data.get("section", chunk.section_context),
                            quoted_text=policy_data.get("quoted_text", "")[:500],
                        )
                    ],
                    parent_id=policy_data.get("parent_id"),
                    children=[],
                    confidence_score=policy_data.get("confidence_score", 0.7),
                )

                policies.append(policy)
                logger.debug(f"Successfully parsed policy {idx + 1}/{len(raw_policies)}: {policy_id} (level {policy.level}, {len(conditions)} conditions)")

            except Exception as e:
                parse_errors += 1
                logger.error(f"Error parsing policy {idx + 1}/{len(raw_policies)}: {e}")
                continue

        if parse_errors > 0:
            logger.warning(f"Failed to parse {parse_errors}/{len(raw_policies)} policies from chunk {chunk.chunk_id}")

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

        # Detect and fix duplicate policy IDs
        logger.debug("Checking for duplicate policy IDs...")
        policies = self._deduplicate_policy_ids(policies)

        # Create policy lookup by ID
        logger.debug("Creating policy lookup map...")
        policy_map = {p.policy_id: p for p in policies}

        # Build parent-child relationships
        logger.debug("Building parent-child relationships...")
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

        logger.info(f"Hierarchy structure - Root policies: {len(root_policies)}, Child policies: {child_count}")

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
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Error extracting additional context: {e}")
            return {}
