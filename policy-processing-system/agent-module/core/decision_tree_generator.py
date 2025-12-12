"""
Decision tree generation system for converting policies into eligibility questions.
"""
import json
import uuid
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from settings import settings
from utils.logger import get_logger
from models.schemas import (
    SubPolicy,
    DecisionTree,
    DecisionNode,
    EligibilityQuestion,
    QuestionOption,
    QuestionType,
    SourceReference,
    PolicyHierarchy,
)
from core.policy_aggregator import PolicyAggregator, PolicyGenerationPlan
from core.tree_generation_prompts import (
    DECISION_TREE_SYSTEM_PROMPT,
    get_aggregator_prompt,
    get_leaf_prompt,
)
from core.tree_validator import validate_tree_structure

logger = get_logger(__name__)


class DecisionTreeGenerator:
    """Generates decision trees from policies with eligibility questions."""

    def __init__(self, use_gpt4: bool = False, llm: Optional[ChatOpenAI] = None, max_tokens: int = 8000):
        """
        Initialize decision tree generator.

        Args:
            use_gpt4: Whether to use GPT-4 for generation
            llm: Optional pre-configured LLM client
            max_tokens: Maximum tokens for LLM response (default 8000 for large trees)
        """
        from utils.llm import get_llm
        
        # If LLM provided, use it; otherwise create with increased token limit
        self.llm = llm if llm is not None else get_llm(use_gpt4=use_gpt4, max_tokens=max_tokens)
        self.use_gpt4 = use_gpt4
        self.max_tokens = max_tokens
        self.aggregator = PolicyAggregator()
        self.generation_plan: Optional[PolicyGenerationPlan] = None
        self.generated_trees: Dict[str, DecisionTree] = {}  # policy_id -> tree
        model_name = "gpt-4o" if use_gpt4 else "gpt-4o-mini"
        logger.info(f"Decision tree generator initialized with model: {model_name} (GPT-4: {use_gpt4}), max_tokens: {max_tokens}")
        logger.debug(f"Configuration - Max concurrent: {settings.openai_max_concurrent_requests}, Request timeout: {settings.openai_per_request_timeout}s")

    async def generate_hierarchical_trees(
        self, policy_hierarchy: PolicyHierarchy
    ) -> List[DecisionTree]:
        """
        Generate contextually-aware decision trees respecting policy hierarchy.

        Args:
            policy_hierarchy: Complete policy hierarchy

        Returns:
            List of DecisionTree objects with hierarchical context
        """
        import asyncio

        logger.info(f"Starting hierarchical decision tree generation for {policy_hierarchy.total_policies} policies...")

        # Analyze hierarchy
        logger.debug("Analyzing policy hierarchy structure...")
        self.generation_plan = self.aggregator.analyze_hierarchy(policy_hierarchy)

        # Get generation order (leaf policies first, then aggregators)
        logger.debug("Determining optimal generation order...")
        ordered_policies = self.aggregator.get_generation_order(self.generation_plan)

        logger.info(
            f"Generation plan - Leaf policies: {len(self.generation_plan.leaf_policies)}, "
            f"Aggregator policies: {len(self.generation_plan.aggregator_policies)}, "
            f"Total to generate: {len(ordered_policies)}"
        )

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(settings.openai_max_concurrent_requests)

        async def generate_with_semaphore(policy: SubPolicy, index: int, total: int):
            async with semaphore:
                return await self._generate_hierarchical_tree(policy, index, total)

        # Filter to only leaf policies (aggregator policies don't need decision trees)
        leaf_policies = [p for p in ordered_policies if len(p.children) == 0]
        logger.info(
            f"Filtered {len(leaf_policies)} leaf policies from {len(ordered_policies)} total policies. "
            f"Skipping {len(ordered_policies) - len(leaf_policies)} aggregator policies (they don't need decision trees)."
        )

        # Generate trees only for leaf policies
        logger.info(f"Creating {len(leaf_policies)} tree generation tasks (leaf policies only)...")
        tasks = []
        for i, policy in enumerate(leaf_policies):
            tasks.append(generate_with_semaphore(policy, i + 1, len(leaf_policies)))

        logger.info(f"Executing tree generation tasks in parallel (max {settings.openai_max_concurrent_requests} concurrent)...")
        # Run all tasks with concurrency control
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful trees
        logger.info("Processing tree generation results...")
        trees = []
        errors_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors_count += 1
                logger.error(f"Error generating tree for policy {leaf_policies[i].policy_id}: {result}")
            elif result is not None:
                trees.append(result)
                self.generated_trees[result.policy_id] = result
                logger.debug(f"Successfully stored tree for policy {result.policy_id}")

        logger.info(f"Tree generation complete - Success: {len(trees)}/{len(leaf_policies)}, Failed: {errors_count}")

        return trees

    async def _generate_hierarchical_tree(
        self, policy: SubPolicy, index: int, total: int
    ) -> Optional[DecisionTree]:
        """
        Generate a contextually-aware tree for a policy.

        Args:
            policy: SubPolicy object
            index: Current policy index
            total: Total number of policies

        Returns:
            DecisionTree with hierarchical context
        """
        import asyncio

        try:
            # Determine if this is a leaf or aggregator policy
            is_aggregator = len(policy.children) > 0
            tree_type = "aggregator" if is_aggregator else "leaf"

            logger.info(f"[Tree {index}/{total}] Starting generation for {tree_type} policy: {policy.policy_id} - {policy.title}")
            logger.debug(f"[Tree {index}/{total}] Policy details - Level: {policy.level}, Children: {len(policy.children)}, Conditions: {len(policy.conditions)}")

            # Generate tree with timeout (parser is now lenient and won't raise ValueError)
            if is_aggregator:
                # Generate aggregator tree that routes to children
                logger.debug(f"[Tree {index}/{total}] Generating aggregator tree with {len(policy.children)} child policies")
                tree = await asyncio.wait_for(
                    self._generate_aggregator_tree(policy),
                    timeout=settings.openai_per_request_timeout
                )
            else:
                # Generate leaf tree with parent context
                logger.debug(f"[Tree {index}/{total}] Generating leaf tree (parent: {policy.parent_id if policy.parent_id else 'None'})")
                tree = await asyncio.wait_for(
                    self._generate_leaf_tree(policy),
                    timeout=settings.openai_per_request_timeout
                )

            logger.info(f"[Tree {index}/{total}] Generation complete - {policy.policy_id} ({tree.total_nodes} nodes, {tree.total_paths} paths, confidence: {tree.confidence_score:.2f})")
            return tree

        except asyncio.TimeoutError:
            logger.error(
                f"[Tree {index}/{total}] Timeout after {settings.openai_per_request_timeout}s - {policy.policy_id}: {policy.title}"
            )
            return None

        except Exception as e:
            logger.error(
                f"[Tree {index}/{total}] Error generating tree - {policy.policy_id}: {policy.title} - {e}",
                exc_info=True
            )
            return None

    async def generate_trees(self, policies: List[SubPolicy]) -> List[DecisionTree]:
        """
        Generate decision trees for all policies with rate limit control.

        Args:
            policies: List of SubPolicy objects

        Returns:
            List of DecisionTree objects
        """
        import asyncio

        # Create semaphore to limit concurrent API requests
        semaphore = asyncio.Semaphore(settings.openai_max_concurrent_requests)

        logger.info(
            f"Generating decision trees for {len(policies)} policies "
            f"(max {settings.openai_max_concurrent_requests} concurrent)..."
        )

        async def generate_with_semaphore(policy: SubPolicy, index: int, total: int):
            async with semaphore:
                return await self._generate_tree_with_error_handling(policy, index, total)

        # Create tasks for all policies
        tasks = []
        for i, policy in enumerate(policies):
            tasks.append(generate_with_semaphore(policy, i + 1, len(policies)))

        # Run all tasks with concurrency control
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and collect successful trees
        trees = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error generating tree for policy {policies[i].policy_id}: {result}")
            elif result is not None:
                trees.append(result)

        logger.info(f"Successfully generated {len(trees)} decision trees")

        return trees

    async def _generate_tree_with_error_handling(
        self, policy: SubPolicy, index: int, total: int
    ) -> Optional[DecisionTree]:
        """
        Generate tree for a policy with error handling and timeout.

        Args:
            policy: SubPolicy object
            index: Current policy index
            total: Total number of policies

        Returns:
            DecisionTree object or None if error
        """
        import asyncio

        try:
            logger.info(f"Generating tree for policy {index}/{total}: {policy.title}")

            # Add timeout per tree
            tree = await asyncio.wait_for(
                self.generate_tree_for_policy(policy),
                timeout=settings.openai_per_request_timeout
            )

            logger.info(f"Completed tree for policy {index}/{total}: {policy.title}")
            return tree

        except asyncio.TimeoutError:
            logger.error(
                f"Timeout generating tree for policy {index}/{total}: {policy.title} "
                f"(exceeded {settings.openai_per_request_timeout} seconds)"
            )
            return None

        except Exception as e:
            logger.error(
                f"Error generating tree for policy {index}/{total}: {policy.title} - {e}",
                exc_info=True
            )
            return None

    @retry(
        stop=stop_after_attempt(settings.openai_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate_tree_for_policy(self, policy: SubPolicy) -> DecisionTree:
        """
        Generate a decision tree for a single policy with validation.

        Args:
            policy: SubPolicy object

        Returns:
            DecisionTree object

        Raises:
            ValueError: If LLM returns invalid or empty response
        """
        logger.debug(f"Creating tree generation prompt for policy {policy.policy_id}...")
        prompt = self._create_tree_generation_prompt(policy)
        prompt_length = len(prompt)

        try:
            if self.llm is None:
                logger.error(f"LLM not initialized for policy {policy.policy_id}")
                raise ValueError(f"LLM not initialized")

            logger.debug(f"Calling LLM API for tree generation (policy: {policy.policy_id}, prompt: {prompt_length} chars)")

            response = await self.llm.ainvoke(prompt)

            logger.debug(f"Received tree response from LLM for policy {policy.policy_id}")

            content = response.content
            logger.debug(f"[LLM Response] Content length: {len(content) if content else 0} characters")

            # Validate response is not empty
            if not content or not content.strip():
                logger.error(f"[LLM Response] Empty response for policy {policy.policy_id}")
                raise ValueError(f"LLM returned empty response for policy {policy.policy_id}")

            # Log first 200 chars for debugging
            logger.debug(f"[LLM Response Preview] {content[:200]}...")

            # Remove explanatory text before JSON
            # LLM often adds "Based on...", "Here is...", etc. before the actual JSON
            if "```json" in content:
                # Extract content between ```json and ```
                parts = content.split("```json")
                if len(parts) > 1:
                    content = parts[1].split("```")[0].strip()
            elif "```" in content:
                # Handle generic code blocks
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            elif content.strip().startswith("{"):
                # Already valid JSON, just strip whitespace
                content = content.strip()
            else:
                # Try to find JSON object in response (starts with {, ends with })
                json_start = content.find("{")
                if json_start != -1:
                    logger.debug(f"[LLM Response] Stripping {json_start} chars of explanatory text before JSON")
                    content = content[json_start:]
                
            # Validate content before JSON parsing
            if not content or not content.strip():
                logger.error(f"[LLM Response] Content empty after cleanup for policy {policy.policy_id}")
                raise ValueError(f"LLM returned empty content after cleanup for policy {policy.policy_id}")

            logger.debug(f"Parsing JSON response for policy {policy.policy_id}...")
            try:
                # STEP 1: Attempt to repair common JSON issues before parsing
                content = self._repair_json(content, policy.policy_id)
                
                # STEP 2: Parse the JSON
                result = json.loads(content)
            except json.JSONDecodeError as e:
                # Log the problematic content
                logger.error(f"[JSON Parse Error] Policy {policy.policy_id}: {e}")
                logger.error(f"[JSON Parse Error] Content (first 500 chars): {content[:500]}")
                logger.error(f"[JSON Parse Error] Error location: line {e.lineno}, column {e.colno}, char {e.pos}")
                
                # Try one more repair attempt for specific common issues
                try:
                    logger.warning(f"[JSON Repair] Attempting emergency repair for policy {policy.policy_id}")
                    content_repaired = self._emergency_json_repair(content)
                    result = json.loads(content_repaired)
                    logger.info(f"[JSON Repair] Successfully repaired JSON for policy {policy.policy_id}")
                except Exception as repair_err:
                    logger.error(f"[JSON Repair] Emergency repair failed: {repair_err}")
                    raise e  # Raise original error
                
            logger.debug(f"[LLM Response] Parsed JSON keys: {list(result.keys())}")

            # Validate result structure before parsing
            if not result or "root_node" not in result:
                logger.error(f"[LLM Response] Missing 'root_node' key. Available keys: {list(result.keys())}")
                raise ValueError(f"LLM response missing 'root_node' for policy {policy.policy_id}")

            root_node_data = result["root_node"]
            logger.debug(f"[LLM Response] Root node type: {type(root_node_data)}, keys: {list(root_node_data.keys()) if isinstance(root_node_data, dict) else 'N/A'}")
            if not root_node_data or not isinstance(root_node_data, dict):
                logger.error(f"[LLM Response] Invalid root_node: {root_node_data}")
                raise ValueError(f"LLM returned empty or invalid root_node for policy {policy.policy_id}")

            # Check for placeholder/stub content
            if root_node_data.get("node_type") == "outcome":
                outcome_text = root_node_data.get("outcome", "").lower()
                if not outcome_text or outcome_text in ["todo", "placeholder", "null", "none", ""]:
                    raise ValueError(f"LLM returned placeholder outcome for policy {policy.policy_id}")

            # Parse result into DecisionTree object
            logger.debug(f"Building decision tree structure for policy {policy.policy_id}...")
            tree = self._parse_tree_result(result, policy)

            # Set hierarchical metadata (IMPORTANT: fixes parent_mismatch warnings)
            tree.policy_level = policy.level
            tree.parent_policy_id = policy.parent_id
            tree.is_aggregator = len(policy.children) > 0 if hasattr(policy, 'children') and policy.children else False

            # Add parent context to root node
            if tree.root_node and policy.parent_id:
                tree.root_node.parent_policy_id = policy.parent_id

            # Calculate statistics
            logger.debug(f"Calculating tree statistics for policy {policy.policy_id}...")
            tree.total_nodes = self._count_nodes(tree.root_node)
            tree.total_paths = self._count_paths(tree.root_node)
            tree.max_depth = self._calculate_depth(tree.root_node)

            # Validate tree structure and routing
            logger.debug(f"Validating tree structure for policy {policy.policy_id}...")
            validation_result = validate_tree_structure(tree)

            # Populate validation fields
            tree.has_complete_routing = validation_result["has_complete_routing"]
            tree.unreachable_nodes = validation_result["unreachable_nodes"]
            tree.incomplete_routes = validation_result["incomplete_routes"]
            tree.total_outcomes = validation_result["outcome_nodes"]

            if not tree.has_complete_routing:
                logger.warning(
                    f"Tree for policy {policy.policy_id} has incomplete routing - "
                    f"Unreachable: {len(tree.unreachable_nodes)}, Incomplete routes: {len(tree.incomplete_routes)}"
                )
            else:
                logger.debug(f"Tree validation passed - all paths complete for policy {policy.policy_id}")

            # Validate tree has meaningful content
            if tree.total_nodes < 2:  # At least question + outcome
                logger.warning(f"Tree for policy {policy.policy_id} has only {tree.total_nodes} nodes - may be incomplete")

            logger.info(
                f"Successfully generated tree for policy {policy.policy_id}: "
                f"{tree.total_nodes} nodes, {tree.total_paths} paths, depth {tree.max_depth}, "
                f"confidence {tree.confidence_score:.2f}, routing_complete: {tree.has_complete_routing}"
            )

            return tree

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for policy {policy.policy_id}: {e}")
            raise ValueError(f"LLM returned invalid JSON for policy {policy.policy_id}: {e}")
        except Exception as e:
            logger.error(f"Error generating tree for policy {policy.policy_id}: {e}")
            raise

    @retry(
        stop=stop_after_attempt(settings.openai_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _generate_aggregator_tree(self, policy: SubPolicy) -> DecisionTree:
        """
        Generate an aggregator tree that routes to child policy trees with validation.

        Args:
            policy: Aggregator policy with children

        Returns:
            DecisionTree that orchestrates child policies

        Raises:
            ValueError: If LLM returns invalid or empty response
        """
        # Get context
        context = self.aggregator.get_policy_context(policy, self.generation_plan)
        aggregation_strategy = self.aggregator.determine_aggregation_strategy(
            policy, self.generation_plan
        )

        # Create specialized prompt for aggregator trees using template
        prompt = get_aggregator_prompt(
            policy_title=policy.title,
            policy_description=policy.description,
            child_policies=policy.children,
            context=str(context)
        )

        try:
            if self.llm is None:
                logger.error(f"LLM not initialized for aggregator policy {policy.policy_id}")
                raise ValueError(f"LLM not initialized")

            response = await self.llm.ainvoke(prompt)

            content = response.content

            # Validate response
            if not content or not content.strip():
                raise ValueError(f"LLM returned empty response for aggregator policy {policy.policy_id}")

            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)

            if not result or "root_node" not in result:
                raise ValueError(f"LLM response missing 'root_node' for aggregator policy {policy.policy_id}")

            tree = self._parse_tree_result(result, policy)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for aggregator policy {policy.policy_id}: {e}")
            raise ValueError(f"LLM returned invalid JSON for aggregator policy {policy.policy_id}: {e}")
        except Exception as e:
            logger.error(f"Error generating aggregator tree for policy {policy.policy_id}: {e}")
            raise

        # Set hierarchical metadata
        tree.policy_level = policy.level
        tree.parent_policy_id = policy.parent_id
        tree.child_policy_ids = [child.policy_id for child in policy.children]
        tree.is_aggregator = True
        tree.aggregation_strategy = aggregation_strategy

        # Add child policy references to root node
        if tree.root_node:
            tree.root_node.child_policy_references = tree.child_policy_ids
            tree.root_node.parent_policy_id = policy.parent_id
            tree.root_node.policy_context = context

        # Calculate statistics
        tree.total_nodes = self._count_nodes(tree.root_node)
        tree.total_paths = self._count_paths(tree.root_node)
        tree.max_depth = self._calculate_depth(tree.root_node)

        # Validate tree structure and routing
        logger.debug(f"Validating aggregator tree structure for policy {policy.policy_id}...")
        validation_result = validate_tree_structure(tree)

        # Populate validation fields
        tree.has_complete_routing = validation_result["has_complete_routing"]
        tree.unreachable_nodes = validation_result["unreachable_nodes"]
        tree.incomplete_routes = validation_result["incomplete_routes"]
        tree.total_outcomes = validation_result["outcome_nodes"]

        if not tree.has_complete_routing:
            logger.warning(
                f"Aggregator tree for policy {policy.policy_id} has incomplete routing - "
                f"Unreachable: {len(tree.unreachable_nodes)}, Incomplete routes: {len(tree.incomplete_routes)}"
            )

        return tree

    @retry(
        stop=stop_after_attempt(settings.openai_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _generate_leaf_tree(self, policy: SubPolicy) -> DecisionTree:
        """
        Generate a leaf tree with parent context.

        Args:
            policy: Leaf policy without children

        Returns:
            DecisionTree with parent context
        """
        # Get context
        context = self.aggregator.get_policy_context(policy, self.generation_plan)

        # Create specialized prompt for leaf trees using template
        parent_context_str = context.get("navigation_hint", "")
        prompt = get_leaf_prompt(
            policy_title=policy.title,
            policy_description=policy.description,
            policy_level=policy.level,
            conditions=policy.conditions,
            parent_context=parent_context_str,
            context=str(context)
        )

        if self.llm is None:
            logger.error(f"LLM not initialized for leaf policy {policy.policy_id}")
            raise ValueError(f"LLM not initialized")

        response = await self.llm.ainvoke(prompt)

        content = response.content
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)
        tree = self._parse_tree_result(result, policy)

        # Set hierarchical metadata
        tree.policy_level = policy.level
        tree.parent_policy_id = policy.parent_id
        tree.child_policy_ids = []
        tree.is_aggregator = False
        tree.aggregation_strategy = None

        # Add parent context to root node
        if tree.root_node:
            tree.root_node.parent_policy_id = policy.parent_id
            tree.root_node.policy_context = context
            if context.get("navigation_hint"):
                tree.root_node.navigation_hint = context["navigation_hint"]

        # Calculate statistics
        tree.total_nodes = self._count_nodes(tree.root_node)
        tree.total_paths = self._count_paths(tree.root_node)
        tree.max_depth = self._calculate_depth(tree.root_node)

        # Validate tree structure and routing
        logger.debug(f"Validating leaf tree structure for policy {policy.policy_id}...")
        validation_result = validate_tree_structure(tree)

        # Populate validation fields
        tree.has_complete_routing = validation_result["has_complete_routing"]
        tree.unreachable_nodes = validation_result["unreachable_nodes"]
        tree.incomplete_routes = validation_result["incomplete_routes"]
        tree.total_outcomes = validation_result["outcome_nodes"]

        if not tree.has_complete_routing:
            logger.warning(
                f"Leaf tree for policy {policy.policy_id} has incomplete routing - "
                f"Unreachable: {len(tree.unreachable_nodes)}, Incomplete routes: {len(tree.incomplete_routes)}"
            )

        return tree

    def _create_aggregator_prompt(
        self, policy: SubPolicy, context: Dict[str, Any], aggregation_strategy: str
    ) -> str:
        """Create prompt for aggregator tree generation."""
        # Serialize child policies
        children_text = "\n".join([
            f"{i+1}. {child.title}\n   Description: {child.description[:200]}..."
            for i, child in enumerate(policy.children)
        ])

        strategy_instructions = {
            "scenario_routing": """
Create a routing tree that asks the user which scenario applies to them, then directs them to the appropriate sub-policy.

The root question should be a multiple_choice question with options for each child policy. Each option should:
- Have a clear, descriptive label based on the child policy title
- Include the "routes_to_tree" field with the child policy_id
- Include the "routes_to_policy" field with the child policy title
- Lead to an outcome node that indicates "Refer to [Child Policy Title]"

Example structure:
{{
  "root_node": {{
    "node_type": "question",
    "question": {{
      "question_type": "multiple_choice",
      "question_text": "Which scenario applies to you?",
      "options": [
        {{
          "option_id": "opt1",
          "label": "[Child Policy 1 Title]",
          "value": "child_1",
          "routes_to_tree": "[child_policy_id_1]",
          "routes_to_policy": "[Child Policy 1 Title]"
        }}
      ]
    }},
    "children": {{
      "[child_1]": {{
        "node_type": "outcome",
        "outcome": "Refer to [Child Policy 1 Title] for detailed eligibility criteria",
        "outcome_type": "refer_to_manual"
      }}
    }}
  }}
}}
""",
            "sequential": "Create a tree that asks initial qualifying questions, then routes to the appropriate child policy based on answers.",
            "conditional": "Create a tree with conditional logic that determines which child policy applies based on specific conditions."
        }

        prompt = f"""Create a navigation/routing decision tree for a parent policy that has multiple sub-policies.

Parent Policy: {policy.title}
Description: {policy.description}
Level: {policy.level}

This policy has {len(policy.children)} sub-policies:
{children_text}

Context:
- Policy hierarchy level: {context.get('level', 0)}
- Has {context.get('child_count', 0)} child policies
- Aggregation strategy: {aggregation_strategy}

{strategy_instructions.get(aggregation_strategy, strategy_instructions['scenario_routing'])}

IMPORTANT: This is a routing tree. The purpose is to help users navigate to the correct sub-policy, NOT to answer all eligibility questions. Keep it simple and focused on scenario identification.

Return a JSON object with the decision tree structure."""

        return prompt

    def _create_leaf_prompt(
        self, policy: SubPolicy, context: Dict[str, Any]
    ) -> str:
        """Create prompt for leaf tree generation with parent context."""
        # Serialize conditions
        conditions_text = "\n".join([
            f"- {cond.description} (Logic: {cond.logic_type})"
            for cond in policy.conditions
        ])

        # Serialize source references
        sources_text = "\n".join([
            f"- Page {ref.page_number}, Section: {ref.section}\n  Quote: {ref.quoted_text[:200]}..."
            for ref in policy.source_references
        ])

        # Build context information
        context_text = f"\nHierarchical Context:\n"
        if context.get("parent_policy_title"):
            context_text += f"- This is a sub-policy of: {context['parent_policy_title']}\n"
            context_text += f"- Context: {context.get('navigation_hint', 'N/A')}\n"
        context_text += f"- Policy hierarchy level: {context.get('level', 0)}\n"

        prompt = f"""Create a comprehensive decision tree for the following policy.

Policy: {policy.title}
Description: {policy.description}
Level: {policy.level}

{context_text}

Conditions:
{conditions_text if conditions_text else "No explicit conditions defined - infer from description"}

Source References:
{sources_text}

Instructions:
1. Convert ALL policy conditions into clear eligibility questions
2. For each question, determine the most appropriate question type:
   - yes_no: Simple yes/no questions
   - multiple_choice: Questions with 2-4 predefined options
   - numeric_range: Questions about ages, amounts, durations
   - text_input: Questions requiring text input (names, IDs)
   - date: Questions about specific dates
   - conditional: Complex questions with sub-conditions

3. Create a logical flow that:
   - Starts with the most important/filtering questions first
   - Asks questions in the most efficient order
   - Handles all possible answer combinations
   - Leads to clear outcomes (approved, denied, refer_to_manual)

4. For each question:
   - Provide clear, user-friendly question text
   - Include explanation of why this question matters
   - Reference the source policy section
   - Add help text if needed
   - Specify validation rules for inputs

5. Ensure the tree handles edge cases and complex logic (AND/OR conditions)

6. Assign confidence scores based on:
   - Clarity of source policy
   - Completeness of logic
   - Potential for ambiguity

IMPORTANT: This is a detailed eligibility tree. Ask all necessary questions to determine if the applicant meets the criteria for THIS SPECIFIC sub-policy.

Return a JSON object with the decision tree structure."""

        return prompt

    def _create_tree_generation_prompt(self, policy: SubPolicy) -> str:
        """
        Create prompt for tree generation (fallback method - uses prompts via get_leaf_prompt).

        Args:
            policy: SubPolicy object

        Returns:
            Prompt string
        """
        # Use the leaf prompt template instead of old prompt
        # This ensures consistency with hierarchical generation
        return get_leaf_prompt(
            policy_title=policy.title,
            policy_description=policy.description,
            policy_level=policy.level,
            conditions=policy.conditions,
            parent_context="",
            context=f"Policy Level: {policy.level}\nConditions: {len(policy.conditions)}"
        )

    def _parse_tree_result(self, result: Dict[str, Any], policy: SubPolicy) -> DecisionTree:
        """
        Parse tree generation result.

        Args:
            result: Generation result dictionary
            policy: Source policy

        Returns:
            DecisionTree object
        """
        logger.debug(f"[_parse_tree_result] Parsing tree result for policy: {policy.policy_id}")
        root_node_data = result.get("root_node", {})
        logger.debug(f"[_parse_tree_result] Root node data keys: {list(root_node_data.keys()) if root_node_data else 'None'}")

        # Parse the root node recursively
        logger.debug(f"[_parse_tree_result] Parsing root node...")
        root_node = self._parse_node(root_node_data)
        logger.debug(f"[_parse_tree_result] Root node parsed - type: {root_node.node_type}, has_question: {root_node.question is not None}, children: {len(root_node.children)}")

        # Extract all questions from the tree into a flat list
        logger.debug(f"[_parse_tree_result] Extracting questions from tree...")
        questions = self._extract_questions(root_node)
        logger.info(f"[_parse_tree_result] Extracted {len(questions)} questions from tree for policy {policy.policy_id}")

        # Create tree
        tree = DecisionTree(
            tree_id=str(uuid.uuid4()),
            policy_id=policy.policy_id,
            policy_title=policy.title,
            root_node=root_node,
            questions=questions,  # Populate the questions list
            total_nodes=0,  # Will be calculated
            total_paths=0,  # Will be calculated
            max_depth=0,  # Will be calculated
            confidence_score=root_node.confidence_score,
        )

        logger.debug(f"[_parse_tree_result] DecisionTree created with {len(tree.questions)} questions")
        return tree

    def _parse_node(self, node_data: Dict[str, Any]) -> DecisionNode:
        """
        Parse a decision node recursively with lenient handling of malformed data.

        Args:
            node_data: Node data dictionary

        Returns:
            DecisionNode object
        """
        # Validate node data
        if not node_data:
            logger.warning("[_parse_node] Empty node data received from LLM, creating stub outcome node")
            return DecisionNode(
                node_id=str(uuid.uuid4()),
                node_type="outcome",
                question=None,
                outcome="Error: Empty node data from LLM",
                outcome_type="refer_to_manual",
                children={},
                source_references=[],
                confidence_score=0.3,
            )

        node_type = node_data.get("node_type", "question")
        logger.debug(f"[_parse_node] Parsing node - node_type: {node_type}, node_id: {node_data.get('node_id', 'None')}")

        # Parse question if present
        question = None
        outcome = None
        outcome_type = None

        if node_type == "question":
            if "question" not in node_data or not node_data["question"]:
                logger.warning(
                    f"[_parse_node] Question node missing 'question' field in node {node_data.get('node_id', 'unknown')}. "
                    "Converting to outcome node."
                )
                # Convert to outcome node rather than failing
                node_type = "outcome"
                outcome = "Incomplete policy data - manual review required"
                outcome_type = "refer_to_manual"
            else:
                logger.debug(f"[_parse_node] Parsing question for node {node_data.get('node_id', 'unknown')}")
                question = self._parse_question(node_data["question"])
                logger.debug(f"[_parse_node] Question parsed: '{question.question_text[:50]}...' (type: {question.question_type})")

        # Get outcome data if outcome node
        if node_type == "outcome":
            if outcome is None:  # Not already set from conversion above
                outcome = node_data.get("outcome", "No outcome specified")
                outcome_type = node_data.get("outcome_type", "refer_to_manual")
            logger.debug(f"[_parse_node] Outcome node: {outcome[:50]}... (type: {outcome_type})")

        # Parse children recursively
        children = {}
        for answer, child_data in node_data.get("children", {}).items():
            try:
                children[answer] = self._parse_node(child_data)
            except Exception as e:
                logger.warning(f"Error parsing child node for answer '{answer}': {e}")
                # Create stub outcome node for this child
                children[answer] = DecisionNode(
                    node_id=str(uuid.uuid4()),
                    node_type="outcome",
                    question=None,
                    outcome="Error parsing child node",
                    outcome_type="refer_to_manual",
                    children={},
                    source_references=[],
                    confidence_score=0.3,
                )

        # Parse source references
        source_refs = []
        if "page_number" in node_data:
            source_refs.append(
                SourceReference(
                    page_number=node_data.get("page_number", 1),
                    section=node_data.get("section", ""),
                    quoted_text=node_data.get("quoted_text", "")[:500],
                )
            )

        # Create node
        node = DecisionNode(
            node_id=node_data.get("node_id", str(uuid.uuid4())),
            node_type=node_type,
            question=question,
            outcome=outcome,
            outcome_type=outcome_type,
            children=children,
            source_references=source_refs,
            confidence_score=node_data.get("confidence_score", 0.7),
        )

        return node

    def _parse_question(self, question_data: Dict[str, Any]) -> EligibilityQuestion:
        """
        Parse an eligibility question.

        Args:
            question_data: Question data dictionary

        Returns:
            EligibilityQuestion object
        """
        logger.debug(f"[_parse_question] Parsing question data with keys: {list(question_data.keys())}")
        logger.debug(f"[_parse_question] Question text: '{question_data.get('question_text', 'N/A')[:100]}...'")

        # Parse question type
        question_type_str = question_data.get("question_type", "yes_no")
        try:
            question_type = QuestionType(question_type_str)
            logger.debug(f"[_parse_question] Question type: {question_type}")
        except ValueError:
            logger.warning(f"[_parse_question] Invalid question type '{question_type_str}', defaulting to YES_NO")
            question_type = QuestionType.YES_NO

        # Parse options if multiple choice
        options = None
        if question_type == QuestionType.MULTIPLE_CHOICE and "options" in question_data:
            options = [
                QuestionOption(
                    option_id=opt.get("option_id", str(uuid.uuid4())),
                    label=opt.get("label", ""),
                    value=opt.get("value", ""),
                    leads_to_node=opt.get("leads_to_node"),
                )
                for opt in question_data["options"]
            ]

        # Parse source references
        source_refs = []
        if "page_number" in question_data:
            source_refs.append(
                SourceReference(
                    page_number=question_data.get("page_number", 1),
                    section=question_data.get("section", ""),
                    quoted_text=question_data.get("quoted_text", "")[:500],
                )
            )

        question = EligibilityQuestion(
            question_id=question_data.get("question_id", str(uuid.uuid4())),
            question_text=question_data.get("question_text", ""),
            question_type=question_type,
            options=options,
            validation_rules=question_data.get("validation_rules"),
            help_text=question_data.get("help_text"),
            source_references=source_refs,
            explanation=question_data.get("explanation", ""),
        )

        return question

    def _count_nodes(self, node: DecisionNode) -> int:
        """
        Count total nodes in tree.

        Args:
            node: Root DecisionNode

        Returns:
            Total node count
        """
        count = 1  # Current node

        for child in node.children.values():
            count += self._count_nodes(child)

        return count

    def _count_paths(self, node: DecisionNode) -> int:
        """
        Count total paths in tree.

        Args:
            node: Root DecisionNode

        Returns:
            Total path count
        """
        if node.node_type == "outcome":
            return 1

        if not node.children:
            return 1

        total_paths = 0
        for child in node.children.values():
            total_paths += self._count_paths(child)

        return total_paths

    def _calculate_depth(self, node: DecisionNode, current_depth: int = 0) -> int:
        """
        Calculate maximum depth of tree.

        Args:
            node: Root DecisionNode
            current_depth: Current depth level

        Returns:
            Maximum depth
        """
        if not node.children:
            return current_depth

        max_child_depth = current_depth

        for child in node.children.values():
            child_depth = self._calculate_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth

    def _extract_questions(self, node: DecisionNode, questions: List[EligibilityQuestion] = None) -> List[EligibilityQuestion]:
        """
        Extract all questions from the decision tree into a flat list.

        Args:
            node: Current DecisionNode
            questions: Accumulated questions list

        Returns:
            List of all EligibilityQuestion objects in the tree
        """
        if questions is None:
            questions = []
            logger.debug(f"[_extract_questions] Starting question extraction from tree")

        # If this node has a question, add it to the list
        if node.question is not None:
            questions.append(node.question)
            logger.debug(f"[_extract_questions] Found question: '{node.question.question_text[:50]}...' (type: {node.question.question_type})")

        # Recursively extract questions from all children
        for answer, child_node in node.children.items():
            logger.debug(f"[_extract_questions] Traversing child for answer '{answer}'")
            self._extract_questions(child_node, questions)

        logger.debug(f"[_extract_questions] Extraction complete. Total questions found: {len(questions)}")
        return questions

    def _repair_json(self, content: str, policy_id: str) -> str:
        """
        Repair common JSON issues before parsing.
        
        Common issues:
        1. Trailing commas in objects/arrays
        2. Missing closing brackets
        3. Unescaped quotes in strings
        4. Comments (not valid JSON)
        
        Args:
            content: JSON string to repair
            policy_id: Policy ID for logging
        
        Returns:
            Repaired JSON string
        """
        import re
        
        original_content = content
        repairs_made = []
        
        # Issue 1: Remove trailing commas before closing braces/brackets
        # Pattern: ,\s*} or ,\s*]
        trailing_comma_pattern = r',(\s*[}\]])'
        matches = re.findall(trailing_comma_pattern, content)
        if matches:
            content = re.sub(trailing_comma_pattern, r'\1', content)
            repairs_made.append(f"removed {len(matches)} trailing commas")
        
        # Issue 2: Remove JavaScript-style comments
        # Single-line comments: // ...
        comment_pattern = r'//.*?$'
        matches = re.findall(comment_pattern, content, re.MULTILINE)
        if matches:
            content = re.sub(comment_pattern, '', content, flags=re.MULTILINE)
            repairs_made.append(f"removed {len(matches)} comments")
        
        # Issue 3: Fix common typos in JSON keywords
        content = content.replace('"True"', 'true')
        content = content.replace('"False"', 'false')
        content = content.replace('"None"', 'null')
        content = content.replace('"null"', 'null')
        
        # Issue 4: Remove zero-width characters and other invisible Unicode
        content = content.replace('\u200b', '')  # Zero-width space
        content = content.replace('\ufeff', '')  # Byte order mark
        
        if repairs_made:
            logger.info(f"[JSON Repair] Policy {policy_id}: {', '.join(repairs_made)}")
        
        return content
    
    def _emergency_json_repair(self, content: str) -> str:
        """
        Emergency JSON repair for severely malformed JSON.
        Attempts more aggressive fixes as last resort.
        
        Args:
            content: Malformed JSON string
        
        Returns:
            Repaired JSON string
        """
        import re
        
        # Try to find the last complete valid closing brace
        # Sometimes LLM output is truncated
        brace_count = 0
        last_valid_pos = -1
        
        for i, char in enumerate(content):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    last_valid_pos = i + 1
        
        if last_valid_pos > 0 and last_valid_pos < len(content):
            logger.warning(
                f"[JSON Repair] Truncating content from {len(content)} to {last_valid_pos} chars "
                f"(found complete JSON object)"
            )
            content = content[:last_valid_pos]
        
        # Remove any trailing incomplete content after last }
        if content.rstrip().endswith('}'):
            # Find position of last }
            last_brace = content.rindex('}')
            content = content[:last_brace + 1]
        
        # Try standard repair again
        content = self._repair_json(content, "emergency_repair")
        
        return content

    async def optimize_tree(self, tree: DecisionTree) -> DecisionTree:
        """
        Optimize decision tree by reordering questions for efficiency.

        Args:
            tree: DecisionTree object

        Returns:
            Optimized DecisionTree
        """
        logger.info(f"Optimizing decision tree for policy {tree.policy_id}")

        # This is a placeholder for more sophisticated optimization
        # Could implement:
        # - Reordering questions based on information gain
        # - Combining similar branches
        # - Removing redundant questions

        return tree
