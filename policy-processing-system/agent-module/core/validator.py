"""
Validation system with confidence scoring for policies and decision trees.
"""
from typing import List, Set, Optional
from langchain_openai import ChatOpenAI
from settings import settings
from utils.logger import get_logger
from models.schemas import (
    PolicyHierarchy,
    SubPolicy,
    DecisionTree,
    DecisionNode,
    ValidationResult,
    ValidationIssue,
)

logger = get_logger(__name__)


class Validator:
    """Validates extracted policies and generated decision trees."""

    def __init__(self, use_gpt4: bool = False, llm: Optional[ChatOpenAI] = None):
        """
        Initialize validator.

        Args:
            use_gpt4: Whether to use GPT-4 for validation (unused for now)
            llm: Optional pre-configured LLM client (unused for now)
        """
        self.llm = llm
        self.use_gpt4 = use_gpt4

    async def validate_all(
        self,
        policy_hierarchy: PolicyHierarchy,
        decision_trees: List[DecisionTree],
        original_text: str,
    ) -> ValidationResult:
        """
        Validate all policies and decision trees.

        Args:
            policy_hierarchy: PolicyHierarchy object
            decision_trees: List of DecisionTree objects
            original_text: Original document text

        Returns:
            ValidationResult object
        """
        logger.info("Starting comprehensive validation...")

        issues = []
        recommendations = []
        sections_requiring_gpt4 = []

        # Validate policy hierarchy
        policy_issues = self._validate_policy_hierarchy(policy_hierarchy)
        issues.extend(policy_issues)

        # Validate decision trees
        tree_issues = await self._validate_decision_trees(decision_trees, policy_hierarchy)
        issues.extend(tree_issues)

        # Check completeness
        completeness_score, completeness_issues = self._check_completeness(
            policy_hierarchy, decision_trees
        )
        issues.extend(completeness_issues)

        # Check consistency
        consistency_score, consistency_issues = self._check_consistency(
            policy_hierarchy, decision_trees
        )
        issues.extend(consistency_issues)

        # Check traceability
        traceability_score = self._check_traceability(policy_hierarchy, decision_trees)

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            policy_hierarchy, decision_trees
        )

        # Identify low-confidence sections
        sections_requiring_gpt4 = self._identify_low_confidence_sections(
            policy_hierarchy, decision_trees
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            issues, completeness_score, consistency_score, traceability_score
        )

        # Determine if validation passed
        is_valid = (
            len([i for i in issues if i.severity == "error"]) == 0
            and overall_confidence >= settings.default_confidence_threshold
        )

        result = ValidationResult(
            is_valid=is_valid,
            overall_confidence=overall_confidence,
            issues=issues,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            traceability_score=traceability_score,
            recommendations=recommendations,
            sections_requiring_gpt4=sections_requiring_gpt4,
        )

        logger.info(
            f"Validation complete: valid={is_valid}, confidence={overall_confidence:.2f}, "
            f"issues={len(issues)}"
        )

        return result

    def _validate_policy_hierarchy(self, policy_hierarchy: PolicyHierarchy) -> List[ValidationIssue]:
        """
        Validate policy hierarchy structure.

        Args:
            policy_hierarchy: PolicyHierarchy object

        Returns:
            List of ValidationIssue objects
        """
        issues = []

        # Check if policies were extracted
        if policy_hierarchy.total_policies == 0:
            issues.append(
                ValidationIssue(
                    severity="error",
                    issue_type="missing_policies",
                    description="No policies were extracted from the document",
                    location="policy_hierarchy",
                    suggestion="Review extraction process and document content",
                )
            )
            return issues

        # Validate each policy
        for policy in self._get_all_policies(policy_hierarchy.root_policies):
            # Check for missing required fields
            if not policy.title:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        issue_type="missing_title",
                        description=f"Policy {policy.policy_id} has no title",
                        location=f"policy:{policy.policy_id}",
                        suggestion="Add a descriptive title",
                    )
                )

            if not policy.description:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        issue_type="missing_description",
                        description=f"Policy {policy.policy_id} has no description",
                        location=f"policy:{policy.policy_id}",
                        suggestion="Add a detailed description",
                    )
                )

            # Check source references
            if not policy.source_references:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        issue_type="missing_source_reference",
                        description=f"Policy {policy.policy_id} has no source references",
                        location=f"policy:{policy.policy_id}",
                        suggestion="Add source references with page numbers and quoted text",
                    )
                )

            # Check confidence score
            if policy.confidence_score < settings.default_confidence_threshold:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        issue_type="low_confidence",
                        description=f"Policy {policy.policy_id} has low confidence score: {policy.confidence_score:.2f}",
                        location=f"policy:{policy.policy_id}",
                        suggestion="Consider using GPT-4 for re-extraction",
                    )
                )

            # Validate parent-child relationships
            if policy.parent_id:
                # Check if parent exists
                parent_exists = any(
                    p.policy_id == policy.parent_id
                    for p in self._get_all_policies(policy_hierarchy.root_policies)
                )
                if not parent_exists:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            issue_type="invalid_parent_reference",
                            description=f"Policy {policy.policy_id} references non-existent parent {policy.parent_id}",
                            location=f"policy:{policy.policy_id}",
                            suggestion="Fix parent reference or set to null",
                        )
                    )

        return issues

    async def _validate_decision_trees(
        self, decision_trees: List[DecisionTree], policy_hierarchy: PolicyHierarchy
    ) -> List[ValidationIssue]:
        """
        Validate decision trees.

        Args:
            decision_trees: List of DecisionTree objects
            policy_hierarchy: PolicyHierarchy object

        Returns:
            List of ValidationIssue objects
        """
        issues = []

        for tree in decision_trees:
            # Check if tree has root node
            if not tree.root_node:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        issue_type="missing_root_node",
                        description=f"Decision tree for policy {tree.policy_id} has no root node",
                        location=f"tree:{tree.tree_id}",
                        suggestion="Regenerate decision tree",
                    )
                )
                continue

            # Validate tree structure
            tree_issues = self._validate_tree_structure(tree)
            issues.extend(tree_issues)

            # Check for orphan nodes (nodes not reachable from root)
            orphan_issues = self._check_for_orphan_nodes(tree)
            issues.extend(orphan_issues)

            # Validate question quality
            question_issues = await self._validate_questions(tree)
            issues.extend(question_issues)

        return issues

    def _validate_tree_structure(self, tree: DecisionTree) -> List[ValidationIssue]:
        """
        Validate tree structure for completeness and correctness.

        Args:
            tree: DecisionTree object

        Returns:
            List of ValidationIssue objects
        """
        issues = []

        def validate_node(node: DecisionNode, path: str, path_nodes: Set[str]):
            # Check for circular reference within current path only
            if node.node_id in path_nodes:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        issue_type="circular_reference",
                        description=f"Circular reference detected at node {node.node_id}",
                        location=f"tree:{tree.tree_id}:node:{node.node_id}",
                        suggestion="Fix tree structure to remove circular reference",
                    )
                )
                return

            # Add current node to path
            current_path_nodes = path_nodes.copy()
            current_path_nodes.add(node.node_id)

            # Validate question nodes
            if node.node_type == "question":
                if not node.question:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            issue_type="missing_question",
                            description=f"Question node {node.node_id} has no question",
                            location=f"tree:{tree.tree_id}:node:{node.node_id}",
                            suggestion="Add question to node or change node type",
                        )
                    )

                if not node.children:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            issue_type="dead_end",
                            description=f"Question node {node.node_id} has no children",
                            location=f"tree:{tree.tree_id}:node:{node.node_id}",
                            suggestion="Add outcome nodes for all possible answers",
                        )
                    )

            # Validate outcome nodes
            elif node.node_type == "outcome":
                if not node.outcome:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            issue_type="missing_outcome",
                            description=f"Outcome node {node.node_id} has no outcome",
                            location=f"tree:{tree.tree_id}:node:{node.node_id}",
                            suggestion="Add outcome description",
                        )
                    )

                if node.children:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            issue_type="outcome_with_children",
                            description=f"Outcome node {node.node_id} has children",
                            location=f"tree:{tree.tree_id}:node:{node.node_id}",
                            suggestion="Outcome nodes should be leaf nodes",
                        )
                    )

            # Check source references
            if not node.source_references and node.node_type == "outcome":
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        issue_type="missing_source_reference",
                        description=f"Node {node.node_id} has no source references",
                        location=f"tree:{tree.tree_id}:node:{node.node_id}",
                        suggestion="Add source references to trace back to policy",
                    )
                )

            # Recursively validate children
            for answer, child in node.children.items():
                validate_node(child, f"{path} -> {answer}", current_path_nodes)

        validate_node(tree.root_node, "root", set())

        return issues

    def _check_for_orphan_nodes(self, tree: DecisionTree) -> List[ValidationIssue]:
        """
        Check for nodes not reachable from root.

        Args:
            tree: DecisionTree object

        Returns:
            List of ValidationIssue objects
        """
        # Basic connectivity check - in a full implementation,
        # we'd track all node IDs and ensure they're all reachable
        return []

    async def _validate_questions(self, tree: DecisionTree) -> List[ValidationIssue]:
        """
        Validate question quality and clarity.

        Args:
            tree: DecisionTree object

        Returns:
            List of ValidationIssue objects
        """
        issues = []

        def check_node(node: DecisionNode):
            if node.node_type == "question" and node.question:
                question = node.question

                # Check question text length
                if len(question.question_text) < 10:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            issue_type="short_question",
                            description=f"Question {question.question_id} is very short",
                            location=f"tree:{tree.tree_id}:question:{question.question_id}",
                            suggestion="Expand question for clarity",
                        )
                    )

                # Check for explanation
                if not question.explanation:
                    issues.append(
                        ValidationIssue(
                            severity="info",
                            issue_type="missing_explanation",
                            description=f"Question {question.question_id} has no explanation",
                            location=f"tree:{tree.tree_id}:question:{question.question_id}",
                            suggestion="Add explanation to help users understand",
                        )
                    )

            # Recursively check children
            for child in node.children.values():
                check_node(child)

        check_node(tree.root_node)

        return issues

    def _check_completeness(
        self, policy_hierarchy: PolicyHierarchy, decision_trees: List[DecisionTree]
    ) -> tuple[float, List[ValidationIssue]]:
        """
        Check completeness of extraction.

        Args:
            policy_hierarchy: PolicyHierarchy object
            decision_trees: List of DecisionTree objects

        Returns:
            Tuple of (completeness score, list of issues)
        """
        issues = []

        # Check if every policy has a decision tree
        policy_ids = set(
            p.policy_id for p in self._get_all_policies(policy_hierarchy.root_policies)
        )
        tree_policy_ids = set(t.policy_id for t in decision_trees)

        missing_trees = policy_ids - tree_policy_ids
        for policy_id in missing_trees:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    issue_type="missing_decision_tree",
                    description=f"Policy {policy_id} has no decision tree",
                    location=f"policy:{policy_id}",
                    suggestion="Generate decision tree for this policy",
                )
            )

        # Check hierarchical completeness
        hierarchical_issues = self._check_hierarchical_completeness(
            policy_hierarchy, decision_trees
        )
        issues.extend(hierarchical_issues)

        # Calculate completeness score
        if policy_ids:
            completeness_score = len(tree_policy_ids) / len(policy_ids)
        else:
            completeness_score = 0.0

        return completeness_score, issues

    def _check_hierarchical_completeness(
        self, policy_hierarchy: PolicyHierarchy, decision_trees: List[DecisionTree]
    ) -> List[ValidationIssue]:
        """
        Check hierarchical completeness of decision trees.

        Validates that:
        - Aggregator policies have trees that reference their children
        - Child policies are properly linked to parents
        - Hierarchical relationships are maintained

        Args:
            policy_hierarchy: PolicyHierarchy object
            decision_trees: List of DecisionTree objects

        Returns:
            List of ValidationIssue objects
        """
        issues = []

        # Create tree lookup
        tree_map = {t.policy_id: t for t in decision_trees}

        # Check all policies
        for policy in self._get_all_policies(policy_hierarchy.root_policies):
            tree = tree_map.get(policy.policy_id)

            if not tree:
                continue

            # Check aggregator policies
            if policy.children:
                # This is an aggregator policy - should reference children
                if not tree.is_aggregator:
                    issues.append(
                        ValidationIssue(
                            severity="info",
                            issue_type="aggregator_metadata_missing",
                            description=f"Policy {policy.policy_id} has children but tree is not marked as aggregator",
                            location=f"tree:{tree.tree_id}",
                            suggestion="Set is_aggregator=True for policies with children",
                        )
                    )

                # Check if tree references all child policies
                expected_children = set(child.policy_id for child in policy.children)
                referenced_children = set(tree.child_policy_ids) if tree.child_policy_ids else set()

                missing_children = expected_children - referenced_children
                if missing_children:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            issue_type="missing_child_references",
                            description=f"Aggregator tree for {policy.policy_id} does not reference all child policies",
                            location=f"tree:{tree.tree_id}",
                            suggestion=f"Missing references to: {', '.join(missing_children)}",
                        )
                    )

            # Check leaf policies
            else:
                # This is a leaf policy - should not be marked as aggregator
                if tree.is_aggregator:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            issue_type="incorrect_aggregator_flag",
                            description=f"Policy {policy.policy_id} has no children but tree is marked as aggregator",
                            location=f"tree:{tree.tree_id}",
                            suggestion="Set is_aggregator=False for leaf policies",
                        )
                    )

            # Check parent-child relationship consistency
            if policy.parent_id:
                if tree.parent_policy_id != policy.parent_id:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            issue_type="parent_mismatch",
                            description=f"Tree parent_policy_id mismatch for {policy.policy_id}",
                            location=f"tree:{tree.tree_id}",
                            suggestion=f"Expected parent {policy.parent_id}, got {tree.parent_policy_id}",
                        )
                    )

        return issues

    def _check_consistency(
        self, policy_hierarchy: PolicyHierarchy, decision_trees: List[DecisionTree]
    ) -> tuple[float, List[ValidationIssue]]:
        """
        Check consistency between policies and decision trees.

        Args:
            policy_hierarchy: PolicyHierarchy object
            decision_trees: List of DecisionTree objects

        Returns:
            Tuple of (consistency score, list of issues)
        """
        issues = []
        consistency_scores = []

        # For each policy with a decision tree, check if the tree represents all conditions
        for policy in self._get_all_policies(policy_hierarchy.root_policies):
            tree = next((t for t in decision_trees if t.policy_id == policy.policy_id), None)

            if tree and policy.conditions:
                # Simple heuristic: check if number of questions >= number of conditions
                question_count = self._count_questions(tree.root_node)
                condition_count = len(policy.conditions)

                if question_count < condition_count:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            issue_type="incomplete_tree",
                            description=f"Decision tree for policy {policy.policy_id} may not cover all conditions",
                            location=f"tree:{tree.tree_id}",
                            suggestion=f"Policy has {condition_count} conditions but tree has only {question_count} questions",
                        )
                    )
                    consistency_scores.append(question_count / condition_count)
                else:
                    consistency_scores.append(1.0)

        # Calculate average consistency score
        if consistency_scores:
            consistency_score = sum(consistency_scores) / len(consistency_scores)
        else:
            consistency_score = 1.0

        return consistency_score, issues

    def _check_traceability(
        self, policy_hierarchy: PolicyHierarchy, decision_trees: List[DecisionTree]
    ) -> float:
        """
        Check traceability to source material.

        Args:
            policy_hierarchy: PolicyHierarchy object
            decision_trees: List of DecisionTree objects

        Returns:
            Traceability score
        """
        total_items = 0
        items_with_references = 0

        # Check policies
        for policy in self._get_all_policies(policy_hierarchy.root_policies):
            total_items += 1
            if policy.source_references:
                items_with_references += 1

        # Check decision tree nodes
        for tree in decision_trees:
            def check_node(node: DecisionNode):
                nonlocal total_items, items_with_references
                total_items += 1
                if node.source_references:
                    items_with_references += 1

                for child in node.children.values():
                    check_node(child)

            check_node(tree.root_node)

        if total_items == 0:
            return 0.0

        return items_with_references / total_items

    def _calculate_overall_confidence(
        self, policy_hierarchy: PolicyHierarchy, decision_trees: List[DecisionTree]
    ) -> float:
        """
        Calculate overall confidence score.

        Args:
            policy_hierarchy: PolicyHierarchy object
            decision_trees: List[DecisionTree] objects

        Returns:
            Overall confidence score
        """
        scores = []

        # Collect policy confidence scores
        for policy in self._get_all_policies(policy_hierarchy.root_policies):
            scores.append(policy.confidence_score)

        # Collect tree confidence scores
        for tree in decision_trees:
            scores.append(tree.confidence_score)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def _identify_low_confidence_sections(
        self, policy_hierarchy: PolicyHierarchy, decision_trees: List[DecisionTree]
    ) -> List[str]:
        """
        Identify sections with low confidence that should use GPT-4.

        Args:
            policy_hierarchy: PolicyHierarchy object
            decision_trees: List of DecisionTree objects

        Returns:
            List of section identifiers
        """
        low_confidence_sections = []

        for policy in self._get_all_policies(policy_hierarchy.root_policies):
            if policy.confidence_score < settings.default_confidence_threshold:
                low_confidence_sections.append(f"Policy: {policy.title}")

        for tree in decision_trees:
            if tree.confidence_score < settings.default_confidence_threshold:
                low_confidence_sections.append(f"Decision Tree: {tree.policy_title}")

        return low_confidence_sections

    def _generate_recommendations(
        self,
        issues: List[ValidationIssue],
        completeness_score: float,
        consistency_score: float,
        traceability_score: float,
    ) -> List[str]:
        """
        Generate recommendations based on validation results.

        Args:
            issues: List of ValidationIssue objects
            completeness_score: Completeness score
            consistency_score: Consistency score
            traceability_score: Traceability score

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Count issues by severity
        error_count = len([i for i in issues if i.severity == "error"])
        warning_count = len([i for i in issues if i.severity == "warning"])

        if error_count > 0:
            recommendations.append(
                f"Address {error_count} critical errors before using this output"
            )

        if warning_count > 5:
            recommendations.append(
                f"Review {warning_count} warnings to improve quality"
            )

        if completeness_score < 0.9:
            recommendations.append(
                "Some policies are missing decision trees - consider regenerating"
            )

        if consistency_score < 0.8:
            recommendations.append(
                "Decision trees may not fully represent policy conditions - review for completeness"
            )

        if traceability_score < 0.8:
            recommendations.append(
                "Improve traceability by adding more source references"
            )

        return recommendations

    def _get_all_policies(self, policies: List[SubPolicy]) -> List[SubPolicy]:
        """
        Get all policies recursively.

        Args:
            policies: List of SubPolicy objects

        Returns:
            Flattened list of all policies
        """
        result = []

        for policy in policies:
            result.append(policy)
            if policy.children:
                result.extend(self._get_all_policies(policy.children))

        return result

    def _count_questions(self, node: DecisionNode) -> int:
        """
        Count questions in a tree.

        Args:
            node: Root DecisionNode

        Returns:
            Question count
        """
        count = 1 if node.node_type == "question" else 0

        for child in node.children.values():
            count += self._count_questions(child)

        return count
