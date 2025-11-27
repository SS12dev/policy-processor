"""
Decision tree generation system for converting policies into eligibility questions.
"""
import json
import uuid
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import settings
from app.utils.logger import get_logger
from app.models.schemas import (
    SubPolicy,
    DecisionTree,
    DecisionNode,
    EligibilityQuestion,
    QuestionOption,
    QuestionType,
    SourceReference,
    PolicyHierarchy,
)
from app.core.policy_aggregator import PolicyAggregator, PolicyGenerationPlan
from app.core.tree_generation_prompts import (
    DECISION_TREE_SYSTEM_PROMPT,
    get_aggregator_prompt,
    get_leaf_prompt,
)
from app.core.tree_validator import validate_tree_structure

logger = get_logger(__name__)


class DecisionTreeGenerator:
    """Generates decision trees from policies with eligibility questions."""

    def __init__(self, use_gpt4: bool = False):
        """
        Initialize decision tree generator.

        Args:
            use_gpt4: Whether to use GPT-4 for generation
        """
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model_secondary if use_gpt4 else settings.openai_model_primary
        self.use_gpt4 = use_gpt4
        self.aggregator = PolicyAggregator()
        self.generation_plan: Optional[PolicyGenerationPlan] = None
        self.generated_trees: Dict[str, DecisionTree] = {}  # policy_id -> tree
        logger.info(f"Decision tree generator initialized with model: {self.model} (GPT-4: {use_gpt4})")
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
        Generate a decision tree for a single policy with enhanced validation.

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
            logger.debug(f"Calling {self.model} API for tree generation (policy: {policy.policy_id}, prompt: {prompt_length} chars)")

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": DECISION_TREE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            logger.debug(f"Received tree response from {self.model} for policy {policy.policy_id}")

            content = response.choices[0].message.content
            logger.debug(f"[LLM Response] Content length: {len(content) if content else 0} characters")

            # Validate response is not empty
            if not content or not content.strip():
                logger.error(f"[LLM Response] Empty response for policy {policy.policy_id}")
                raise ValueError(f"LLM returned empty response for policy {policy.policy_id}")

            logger.debug(f"Parsing JSON response for policy {policy.policy_id}...")
            result = json.loads(content)
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
        Generate an aggregator tree that routes to child policy trees with enhanced validation.

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

        # Create specialized prompt for aggregator trees using enhanced template
        prompt = get_aggregator_prompt(
            policy_title=policy.title,
            policy_description=policy.description,
            child_policies=policy.children,
            context=str(context)
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": DECISION_TREE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content

            # Validate response
            if not content or not content.strip():
                raise ValueError(f"LLM returned empty response for aggregator policy {policy.policy_id}")

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

        # Create specialized prompt for leaf trees using enhanced template
        parent_context_str = context.get("navigation_hint", "")
        prompt = get_leaf_prompt(
            policy_title=policy.title,
            policy_description=policy.description,
            policy_level=policy.level,
            conditions=policy.conditions,
            parent_context=parent_context_str,
            context=str(context)
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DECISION_TREE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
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
        Create prompt for tree generation.

        Args:
            policy: SubPolicy object

        Returns:
            Prompt string
        """
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

        prompt = f"""Create a comprehensive decision tree for the following policy.

Policy: {policy.title}
Description: {policy.description}
Level: {policy.level}

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

Return a JSON object with this structure:
{{
  "root_node": {{
    "node_id": "unique_id",
    "node_type": "question",  // or "outcome"
    "question": {{
      "question_id": "q1",
      "question_text": "Are you currently employed?",
      "question_type": "yes_no",
      "explanation": "Employment status affects eligibility per Section 2.1",
      "help_text": "Include full-time, part-time, and self-employment",
      "page_number": 5,
      "section": "Section 2.1",
      "quoted_text": "exact source text"
    }},
    "children": {{
      "yes": {{
        "node_id": "node_2",
        "node_type": "question",
        ...
      }},
      "no": {{
        "node_id": "node_3",
        "node_type": "outcome",
        "outcome": "Not eligible - employment required",
        "outcome_type": "denied",
        "page_number": 5,
        "section": "Section 2.1",
        "quoted_text": "source text",
        "confidence_score": 0.95
      }}
    }},
    "confidence_score": 0.9
  }}
}}

For multiple choice questions, use this format:
{{
  "question_type": "multiple_choice",
  "options": [
    {{
      "option_id": "opt1",
      "label": "Full-time",
      "value": "full_time",
      "leads_to_node": "node_id"
    }},
    ...
  ]
}}

Be thorough and create a complete decision tree that handles ALL scenarios."""

        return prompt

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
