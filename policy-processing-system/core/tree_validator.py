"""
Decision tree validation and path verification utilities.

Validates decision trees for:
- Complete routing (all paths lead to outcomes)
- Reachability (no unreachable nodes)
- Routing rule consistency
- Logic group validity
"""
from typing import Set, List, Dict, Tuple, Optional, Any
from app.models.schemas import (
    DecisionTree,
    DecisionNode,
    RoutingRule,
    LogicGroup,
    NodeType,
    OutcomeType
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TreeValidator:
    """Validates decision tree structure and routing."""

    def __init__(self):
        self.visited_nodes: Set[str] = set()
        self.all_node_ids: Set[str] = set()
        self.outcome_nodes: Set[str] = set()

    def validate_tree(self, tree: DecisionTree) -> Tuple[bool, List[str], List[str]]:
        """
        Validate complete decision tree structure.

        Args:
            tree: DecisionTree to validate

        Returns:
            Tuple of (is_valid, unreachable_nodes, incomplete_routes)
        """
        logger.info(f"Validating decision tree: {tree.tree_id} ({tree.policy_title})")

        self.visited_nodes = set()
        self.all_node_ids = set()
        self.outcome_nodes = set()

        # Collect all node IDs
        self._collect_node_ids(tree.root_node)
        logger.debug(f"Total nodes in tree: {len(self.all_node_ids)}")

        # Validate reachability from root
        self._mark_reachable(tree.root_node)
        unreachable = self.all_node_ids - self.visited_nodes

        if unreachable:
            logger.warning(f"Found {len(unreachable)} unreachable nodes: {list(unreachable)[:5]}")

        # Validate routing completeness
        incomplete_routes = self._find_incomplete_routes(tree.root_node)

        if incomplete_routes:
            logger.warning(f"Found {len(incomplete_routes)} nodes with incomplete routing")

        # Validate logic groups
        if tree.logic_groups:
            self._validate_logic_groups(tree)

        is_valid = len(unreachable) == 0 and len(incomplete_routes) == 0

        logger.info(
            f"Tree validation complete: valid={is_valid}, "
            f"outcomes={len(self.outcome_nodes)}, "
            f"paths={tree.total_paths}"
        )

        return is_valid, list(unreachable), incomplete_routes

    def _collect_node_ids(self, node: DecisionNode) -> None:
        """Recursively collect all node IDs in tree."""
        self.all_node_ids.add(node.node_id)

        if node.node_type == "outcome":
            self.outcome_nodes.add(node.node_id)

        for child in node.children.values():
            self._collect_node_ids(child)

    def _mark_reachable(self, node: DecisionNode) -> None:
        """Mark all reachable nodes from this node."""
        if node.node_id in self.visited_nodes:
            return

        self.visited_nodes.add(node.node_id)

        # Mark children as reachable
        for child in node.children.values():
            self._mark_reachable(child)

        # Also check routing rules for node IDs
        if node.routing_rules:
            for rule in node.routing_rules:
                if rule.next_node_id and rule.next_node_id in self.all_node_ids:
                    target_node = self._find_node_by_id(rule.next_node_id, node)
                    if target_node:
                        self._mark_reachable(target_node)

    def _find_node_by_id(self, node_id: str, search_root: DecisionNode) -> Optional[DecisionNode]:
        """Find a node by ID in the tree."""
        if search_root.node_id == node_id:
            return search_root

        for child in search_root.children.values():
            result = self._find_node_by_id(node_id, child)
            if result:
                return result

        return None

    def _find_incomplete_routes(self, node: DecisionNode, path: List[str] = None) -> List[str]:
        """
        Find nodes with incomplete routing.

        A node has incomplete routing if:
        - It's a question node with no children or routing rules
        - It's a decision node with no default path
        - It has children but some answer values have no route
        """
        if path is None:
            path = []

        incomplete = []

        # Outcome nodes are terminal, always complete
        if node.node_type == "outcome":
            return incomplete

        path = path + [node.node_id]

        # Check if question node has routing
        if node.node_type == "question":
            if not node.children and not node.routing_rules:
                logger.debug(f"Question node {node.node_id} has no children or routing rules")
                incomplete.append(node.node_id)

            # For yes/no questions, ensure both yes and no paths exist
            if node.question and node.question.question_type.value == "yes_no":
                has_yes = "yes" in node.children or (
                    node.routing_rules and any(r.answer_value == "yes" for r in node.routing_rules)
                )
                has_no = "no" in node.children or (
                    node.routing_rules and any(r.answer_value == "no" for r in node.routing_rules)
                )

                if not (has_yes and has_no):
                    logger.debug(f"Yes/No question {node.node_id} missing routes: yes={has_yes}, no={has_no}")
                    incomplete.append(node.node_id)

        # Check if decision node has fallback
        if node.node_type == "decision":
            if not node.children and not node.default_next_node_id:
                logger.debug(f"Decision node {node.node_id} has no children or default route")
                incomplete.append(node.node_id)

        # Recursively check children
        for child in node.children.values():
            incomplete.extend(self._find_incomplete_routes(child, path))

        return incomplete

    def _validate_logic_groups(self, tree: DecisionTree) -> None:
        """Validate logic group structure."""
        for group in tree.logic_groups:
            # Check all member nodes exist
            for node_id in group.member_node_ids:
                if node_id not in self.all_node_ids:
                    logger.warning(
                        f"Logic group {group.group_id} references non-existent node {node_id}"
                    )

            # Check parent group exists if specified
            if group.parent_group_id:
                parent_exists = any(
                    g.group_id == group.parent_group_id for g in tree.logic_groups
                )
                if not parent_exists:
                    logger.warning(
                        f"Logic group {group.group_id} references non-existent parent {group.parent_group_id}"
                    )

    def validate_routing_rule(self, rule: RoutingRule) -> bool:
        """
        Validate a single routing rule.

        Args:
            rule: RoutingRule to validate

        Returns:
            True if valid
        """
        # Check required fields
        if not rule.next_node_id:
            logger.warning("Routing rule missing next_node_id")
            return False

        # Validate range rules
        if rule.comparison.value == "in_range":
            if rule.range_min is None or rule.range_max is None:
                logger.warning(f"Range comparison rule missing range_min or range_max")
                return False

            if rule.range_min > rule.range_max:
                logger.warning(f"Invalid range: min {rule.range_min} > max {rule.range_max}")
                return False

        return True

    def get_all_paths(self, tree: DecisionTree) -> List[List[str]]:
        """
        Get all possible paths through the tree.

        Args:
            tree: DecisionTree to analyze

        Returns:
            List of paths, where each path is a list of node IDs
        """
        paths = []
        self._collect_paths(tree.root_node, [], paths)
        return paths

    def _collect_paths(
        self,
        node: DecisionNode,
        current_path: List[str],
        all_paths: List[List[str]]
    ) -> None:
        """Recursively collect all paths through the tree."""
        current_path = current_path + [node.node_id]

        # If outcome node, path is complete
        if node.node_type == "outcome":
            all_paths.append(current_path)
            return

        # Continue through children
        if node.children:
            for child in node.children.values():
                self._collect_paths(child, current_path, all_paths)
        else:
            # Dead end - incomplete path
            all_paths.append(current_path)

    def get_path_to_outcome(
        self,
        tree: DecisionTree,
        outcome_type: str
    ) -> Optional[List[str]]:
        """
        Find a path to a specific outcome type.

        Args:
            tree: DecisionTree to search
            outcome_type: Outcome type to find (approved, denied, etc.)

        Returns:
            Path to outcome or None
        """
        return self._find_path_to_outcome(tree.root_node, outcome_type, [])

    def _find_path_to_outcome(
        self,
        node: DecisionNode,
        target_outcome: str,
        path: List[str]
    ) -> Optional[List[str]]:
        """Recursively search for path to outcome."""
        path = path + [node.node_id]

        if node.node_type == "outcome" and node.outcome_type == target_outcome:
            return path

        for child in node.children.values():
            result = self._find_path_to_outcome(child, target_outcome, path)
            if result:
                return result

        return None


def validate_tree_structure(tree: DecisionTree) -> Dict[str, Any]:
    """
    Validate tree structure and return detailed report.

    Args:
        tree: DecisionTree to validate

    Returns:
        Validation report dictionary
    """
    validator = TreeValidator()
    is_valid, unreachable, incomplete = validator.validate_tree(tree)

    all_paths = validator.get_all_paths(tree)
    complete_paths = [p for p in all_paths if validator.outcome_nodes and p[-1] in validator.outcome_nodes]

    report = {
        "is_valid": is_valid,
        "total_nodes": len(validator.all_node_ids),
        "reachable_nodes": len(validator.visited_nodes),
        "unreachable_nodes": unreachable,
        "incomplete_routes": incomplete,
        "total_paths": len(all_paths),
        "complete_paths": len(complete_paths),
        "outcome_nodes": len(validator.outcome_nodes),
        "has_complete_routing": len(incomplete) == 0 and len(unreachable) == 0,
    }

    logger.info(f"Tree validation report: {report}")
    return report
