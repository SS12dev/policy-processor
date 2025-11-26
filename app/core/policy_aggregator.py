"""
Policy hierarchy aggregation and analysis.

This module provides policy-agnostic hierarchy analysis for generating
contextually-aware decision trees.
"""
from typing import List, Dict, Set, Tuple, Any, Optional
from app.models.schemas import SubPolicy, PolicyHierarchy
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HierarchyNode:
    """Represents a node in the policy hierarchy analysis."""

    def __init__(self, policy: SubPolicy):
        self.policy = policy
        self.children: List['HierarchyNode'] = []
        self.parent: Optional['HierarchyNode'] = None
        self.depth = 0
        self.is_leaf = True
        self.sibling_count = 0


class PolicyGenerationPlan:
    """Plan for generating decision trees with hierarchical context."""

    def __init__(self):
        self.leaf_policies: List[SubPolicy] = []  # Policies with no children
        self.aggregator_policies: List[SubPolicy] = []  # Policies with children
        self.policy_map: Dict[str, SubPolicy] = {}  # policy_id -> SubPolicy
        self.hierarchy_map: Dict[str, HierarchyNode] = {}  # policy_id -> HierarchyNode
        self.parent_child_map: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        self.child_parent_map: Dict[str, str] = {}  # child_id -> parent_id


class PolicyAggregator:
    """
    Policy-agnostic hierarchy aggregator.

    Analyzes policy structure and creates a generation plan for
    contextually-aware decision trees.
    """

    def __init__(self):
        pass

    def analyze_hierarchy(self, policy_hierarchy: PolicyHierarchy) -> PolicyGenerationPlan:
        """
        Analyze policy hierarchy and create generation plan.

        Args:
            policy_hierarchy: Complete policy hierarchy

        Returns:
            PolicyGenerationPlan with analysis results
        """
        logger.info(f"Analyzing policy hierarchy with {policy_hierarchy.total_policies} policies...")

        plan = PolicyGenerationPlan()

        # Flatten hierarchy into policy map
        self._flatten_hierarchy(policy_hierarchy.root_policies, plan)

        # Build parent-child relationships
        self._build_relationships(plan)

        # Classify policies as leaf or aggregator
        self._classify_policies(plan)

        # Build hierarchy nodes with depth information
        self._build_hierarchy_nodes(policy_hierarchy.root_policies, plan, depth=0, parent=None)

        logger.info(
            f"Hierarchy analysis complete: "
            f"{len(plan.leaf_policies)} leaf policies, "
            f"{len(plan.aggregator_policies)} aggregator policies"
        )

        return plan

    def _flatten_hierarchy(
        self, policies: List[SubPolicy], plan: PolicyGenerationPlan
    ) -> None:
        """Recursively flatten hierarchy into a map."""
        for policy in policies:
            plan.policy_map[policy.policy_id] = policy

            if policy.children:
                self._flatten_hierarchy(policy.children, plan)

    def _build_relationships(self, plan: PolicyGenerationPlan) -> None:
        """Build parent-child relationship maps."""
        for policy_id, policy in plan.policy_map.items():
            # Build parent -> children map
            if policy.children:
                plan.parent_child_map[policy_id] = [child.policy_id for child in policy.children]

            # Build child -> parent map
            if policy.parent_id:
                plan.child_parent_map[policy_id] = policy.parent_id

    def _classify_policies(self, plan: PolicyGenerationPlan) -> None:
        """Classify policies as leaf (no children) or aggregator (has children)."""
        for policy_id, policy in plan.policy_map.items():
            if policy.children:
                # Has children - this is an aggregator
                plan.aggregator_policies.append(policy)
            else:
                # No children - this is a leaf
                plan.leaf_policies.append(policy)

    def _build_hierarchy_nodes(
        self,
        policies: List[SubPolicy],
        plan: PolicyGenerationPlan,
        depth: int,
        parent: Optional[HierarchyNode]
    ) -> None:
        """Build hierarchy nodes with depth and relationship information."""
        for policy in policies:
            node = HierarchyNode(policy)
            node.depth = depth
            node.parent = parent
            node.is_leaf = len(policy.children) == 0

            if parent:
                parent.children.append(node)

            plan.hierarchy_map[policy.policy_id] = node

            # Recursively process children
            if policy.children:
                node.sibling_count = len(policy.children)
                self._build_hierarchy_nodes(
                    policy.children, plan, depth + 1, node
                )

    def get_policy_context(
        self, policy: SubPolicy, plan: PolicyGenerationPlan
    ) -> Dict[str, Any]:
        """
        Get contextual information for a policy.

        Args:
            policy: The policy to get context for
            plan: The generation plan

        Returns:
            Dictionary with context information
        """
        node = plan.hierarchy_map.get(policy.policy_id)
        if not node:
            return {}

        context = {
            "policy_id": policy.policy_id,
            "policy_title": policy.title,
            "level": policy.level,
            "depth": node.depth,
            "is_leaf": node.is_leaf,
            "has_children": len(policy.children) > 0,
            "child_count": len(policy.children)
        }

        # Add parent context
        if node.parent:
            context["parent_policy_id"] = node.parent.policy.policy_id
            context["parent_policy_title"] = node.parent.policy.title
            context["navigation_hint"] = (
                f"This is a sub-policy of '{node.parent.policy.title}'"
            )

        # Add sibling context
        if node.parent:
            context["sibling_count"] = node.parent.sibling_count

        # Add children context
        if policy.children:
            context["child_policies"] = [
                {
                    "policy_id": child.policy_id,
                    "title": child.title,
                    "description": child.description[:100] + "..." if len(child.description) > 100 else child.description
                }
                for child in policy.children
            ]

        return context

    def determine_aggregation_strategy(
        self, policy: SubPolicy, plan: PolicyGenerationPlan
    ) -> Optional[str]:
        """
        Determine how an aggregator policy should route to its children.

        This is policy-agnostic and based on structure, not content.

        Args:
            policy: Aggregator policy
            plan: Generation plan

        Returns:
            Aggregation strategy: 'scenario_routing', 'sequential', 'conditional', None
        """
        if not policy.children:
            return None

        # If policy has multiple children with distinct titles, use scenario routing
        if len(policy.children) >= 2:
            # Check if children have distinct, meaningful titles
            distinct_titles = len(set(child.title for child in policy.children))
            if distinct_titles == len(policy.children):
                return "scenario_routing"  # Route based on which scenario applies

        # If children seem sequential (numbered or ordered), use sequential
        child_ids = [child.policy_id for child in policy.children]
        if self._seems_sequential(child_ids):
            return "sequential"  # Ask questions in order

        # Default to conditional routing
        return "conditional"  # Route based on conditions

    def _seems_sequential(self, policy_ids: List[str]) -> bool:
        """
        Check if policy IDs suggest a sequential structure.

        Examples:
            - section_1_1, section_1_2, section_1_3 -> True
            - step_1, step_2, step_3 -> True
            - policy_a, policy_b, policy_c -> False
        """
        if len(policy_ids) < 2:
            return False

        # Extract numeric suffixes
        numbers = []
        for policy_id in policy_ids:
            parts = policy_id.split('_')
            if parts and parts[-1].isdigit():
                numbers.append(int(parts[-1]))

        # Check if numbers are consecutive
        if len(numbers) == len(policy_ids) and len(numbers) >= 2:
            return numbers == list(range(min(numbers), max(numbers) + 1))

        return False

    def get_generation_order(self, plan: PolicyGenerationPlan) -> List[SubPolicy]:
        """
        Get the order in which to generate decision trees.

        Strategy: Generate leaf policies first (bottom-up), then aggregators.
        This ensures child trees exist when parent trees reference them.

        Args:
            plan: Generation plan

        Returns:
            List of policies in generation order
        """
        # Sort leaf policies by depth (deepest first)
        leaf_policies_sorted = sorted(
            plan.leaf_policies,
            key=lambda p: plan.hierarchy_map[p.policy_id].depth,
            reverse=True
        )

        # Sort aggregator policies by depth (deepest first)
        aggregator_policies_sorted = sorted(
            plan.aggregator_policies,
            key=lambda p: plan.hierarchy_map[p.policy_id].depth,
            reverse=True
        )

        # Return: leaf policies first, then aggregators
        return leaf_policies_sorted + aggregator_policies_sorted
